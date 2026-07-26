"""
MLflow model-registry helpers and the promotion CLI.

    python -m app.registry list
    python -m app.registry promote --version 3 --stage Production

Promotion moves a version between stages and archives whatever previously held
that stage. A running API picks up the new Production version on its next poll,
so no redeploy is needed. Each stage is mirrored to a lowercase alias
(``production`` / ``staging``) so clients can migrate off deprecated stages.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

STAGES = ("Staging", "Production")


@dataclass(frozen=True)
class ModelRef:
    name: str
    version: int
    stage: str
    run_id: str | None


@contextmanager
def _stages_api() -> Iterator[None]:
    # Registry stages are deprecated upstream in favour of aliases; we use them
    # deliberately (see docs/adr/0001) and keep aliases in sync.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*registry stages", category=FutureWarning)
        yield


def _is_missing(exc: MlflowException) -> bool:
    return exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def latest_in_stage(client: MlflowClient, name: str, stage: str) -> ModelRef | None:
    """The newest version currently in ``stage``, or None."""
    try:
        with _stages_api():
            versions = client.get_latest_versions(name, stages=[stage])
    except MlflowException as exc:
        if _is_missing(exc):
            return None
        raise
    if not versions:
        return None
    v = max(versions, key=lambda mv: int(mv.version))
    return ModelRef(name=name, version=int(v.version), stage=stage, run_id=v.run_id)


def promote(client: MlflowClient, name: str, version: int, stage: str, archive_existing: bool = True) -> ModelRef:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    with _stages_api():
        mv = client.transition_model_version_stage(
            name, str(version), stage, archive_existing_versions=archive_existing
        )
    client.set_registered_model_alias(name, stage.lower(), str(version))
    return ModelRef(name=name, version=int(mv.version), stage=stage, run_id=mv.run_id)


def list_versions(client: MlflowClient, name: str) -> list[ModelRef]:
    versions = client.search_model_versions(f"name = '{name}'")
    return sorted(
        (ModelRef(name=name, version=int(v.version), stage=v.current_stage, run_id=v.run_id) for v in versions),
        key=lambda r: r.version,
    )


def main(argv: Sequence[str] | None = None) -> None:
    from app.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Inspect and promote registered SmartLift models")
    parser.add_argument("--model-name", default=settings.model_name)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list registered versions and their stages")
    p = sub.add_parser("promote", help="move a version into a stage")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--keep-existing", action="store_true", help="don't archive versions already in the stage")
    args = parser.parse_args(argv)

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    if args.command == "list":
        for ref in list_versions(client, args.model_name):
            print(f"{ref.name} v{ref.version}  {ref.stage:<10}  run {ref.run_id}")
    else:
        ref = promote(client, args.model_name, args.version, args.stage, archive_existing=not args.keep_existing)
        print(f"{ref.name} v{ref.version} -> {ref.stage}")


if __name__ == "__main__":
    main()
