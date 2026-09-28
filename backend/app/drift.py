"""
Feature drift detection with the Population Stability Index (PSI).

At training time ``compute_reference`` bins every model feature: deciles of the
training data for continuous features, one bin per value for discrete ones
(exercise, RIR, sets, frequency). The reference is logged next to the model in
MLflow, so each registered version carries the distribution it was trained on.

``run_drift_check`` compares that reference with the features of predictions
served by the current Production version over a rolling window:

    PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)

Rule of thumb: < 0.1 stable, 0.1–0.2 moderate shift, > 0.2 significant drift.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import mlflow.artifacts
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.db import DriftResult, InferenceLog
from app.features import FEATURE_COLUMNS
from app.metrics import DRIFT_CHECKS, DRIFT_SAMPLES, FEATURE_PSI

logger = logging.getLogger(__name__)

REFERENCE_ARTIFACT = "drift/reference.json"
MAX_CATEGORIES = 12  # features with at most this many distinct values are binned per value
NUM_QUANTILE_BINS = 10
EPSILON = 1e-4  # floor for empty bins so ln() stays finite


def compute_reference(df: pd.DataFrame, columns: Sequence[str] = FEATURE_COLUMNS) -> dict[str, Any]:
    """Bin spec and expected proportions for each feature in ``df``."""
    features: dict[str, Any] = {}
    for col in columns:
        series = df[col]
        values = sorted(series.dropna().unique().tolist())
        if series.dtype == object or len(values) <= MAX_CATEGORIES:
            spec: dict[str, Any] = {"kind": "categorical", "categories": values}
        else:
            qs = np.quantile(series.astype(float), np.linspace(0, 1, NUM_QUANTILE_BINS + 1)[1:-1])
            spec = {"kind": "numeric", "edges": sorted({float(q) for q in qs})}
        spec["proportions"] = bin_proportions(series.tolist(), spec)
        features[col] = spec
    return {"version": 1, "n_samples": len(df), "features": features}


def bin_proportions(values: Sequence[Any], spec: dict[str, Any]) -> list[float]:
    """Share of ``values`` in each bin of ``spec`` (sums to 1, or all zeros if empty)."""
    if spec["kind"] == "categorical":
        categories = spec["categories"]
        # Last bin collects values never seen in training.
        counts = [0] * (len(categories) + 1)
        index = {c: i for i, c in enumerate(categories)}
        for v in values:
            counts[index.get(v, len(categories))] += 1
    else:
        # Interior edges -> len(edges)+1 bins, open at both ends.
        idx = np.searchsorted(np.asarray(spec["edges"], dtype=float), np.asarray(values, dtype=float), side="right")
        counts = np.bincount(idx, minlength=len(spec["edges"]) + 1).tolist()
    total = sum(counts)
    return [c / total if total else 0.0 for c in counts]


def psi(expected: Sequence[float], actual: Sequence[float], epsilon: float = EPSILON) -> float:
    if len(expected) != len(actual):
        raise ValueError("expected and actual must have the same number of bins")
    total = 0.0
    for e, a in zip(expected, actual, strict=True):
        e, a = max(e, epsilon), max(a, epsilon)
        total += (a - e) * math.log(a / e)
    return total


def feature_psi(reference: dict[str, Any], live: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for col, spec in reference["features"].items():
        if col not in live:
            continue
        expected = list(spec["proportions"])
        if spec["kind"] == "categorical" and len(expected) == len(spec["categories"]):
            expected.append(0.0)  # references always reserve an "unseen" bin; tolerate ones that don't
        out[col] = round(psi(expected, bin_proportions(live[col].tolist(), spec)), 6)
    return out


@dataclass(frozen=True)
class DriftReport:
    status: str  # ok | drift | insufficient_data | no_model | no_reference
    model_version: int | None
    sample_size: int
    psi: dict[str, float]
    drifted: list[str]


def load_reference(run_id: str) -> dict[str, Any] | None:
    try:
        return dict(mlflow.artifacts.load_dict(f"runs:/{run_id}/{REFERENCE_ARTIFACT}"))
    except Exception:
        logger.warning("no drift reference for run %s", run_id, exc_info=True)
        return None


def live_window(db: Session, model_version: int, window: timedelta, max_samples: int, now: datetime) -> pd.DataFrame:
    stmt = (
        select(*(getattr(InferenceLog, c) for c in FEATURE_COLUMNS))
        .where(InferenceLog.model_version == model_version, InferenceLog.created_at >= now - window)
        .order_by(InferenceLog.created_at.desc())
        .limit(max_samples)
    )
    return pd.DataFrame(db.execute(stmt).all(), columns=FEATURE_COLUMNS)


def run_drift_check(
    session_factory: sessionmaker[Session],
    model_version: int | None,
    run_id: str | None,
    *,
    window: timedelta,
    min_samples: int,
    max_samples: int,
    threshold: float,
    now: datetime | None = None,
    reference: dict[str, Any] | None = None,
) -> DriftReport:
    """Compute PSI for every feature, persist results, update gauges, log drift."""
    now = now or datetime.now(UTC)
    if model_version is None or run_id is None:
        DRIFT_CHECKS.labels("no_model").inc()
        return DriftReport("no_model", None, 0, {}, [])

    reference = reference or load_reference(run_id)
    if reference is None:
        DRIFT_CHECKS.labels("no_reference").inc()
        return DriftReport("no_reference", model_version, 0, {}, [])

    with session_factory() as db:
        live = live_window(db, model_version, window, max_samples, now)
        DRIFT_SAMPLES.set(len(live))
        if len(live) < min_samples:
            logger.info(
                "drift check skipped: %d samples < %d",
                len(live),
                min_samples,
                extra={"model_version": model_version, "sample_size": len(live)},
            )
            DRIFT_CHECKS.labels("insufficient_data").inc()
            return DriftReport("insufficient_data", model_version, len(live), {}, [])

        scores = feature_psi(reference, live)
        drifted = sorted(f for f, v in scores.items() if v > threshold)
        for feature, value in scores.items():
            FEATURE_PSI.labels(feature).set(value)
            db.add(
                DriftResult(
                    computed_at=now,
                    model_version=model_version,
                    feature=feature,
                    psi=value,
                    sample_size=len(live),
                    window_start=now - window,
                    window_end=now,
                    threshold=threshold,
                    drifted=value > threshold,
                )
            )
            if value > threshold:
                logger.warning(
                    "feature drift detected: %s PSI %.3f > %.2f",
                    feature,
                    value,
                    threshold,
                    extra={"feature": feature, "psi": value, "threshold": threshold, "model_version": model_version},
                )
        db.commit()

    status = "drift" if drifted else "ok"
    DRIFT_CHECKS.labels(status).inc()
    logger.info(
        "drift check complete",
        extra={"model_version": model_version, "sample_size": len(live), "drifted": drifted, "status": status},
    )
    return DriftReport(status, model_version, len(live), scores, drifted)


def latest_results(db: Session) -> list[DriftResult]:
    """Results of the most recent drift check."""
    latest = db.scalar(select(DriftResult.computed_at).order_by(DriftResult.computed_at.desc()).limit(1))
    if latest is None:
        return []
    return list(db.scalars(select(DriftResult).where(DriftResult.computed_at == latest).order_by(DriftResult.feature)))
