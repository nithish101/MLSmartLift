"""
Performance gate over Locust's CSV statistics.

    python -m loadtest.check_results results/locust_stats.csv --target-rps 50

Exits 1 if any of these are breached for the aggregated (post-warmup) traffic:
  * p99 latency above --max-p99-ms
  * error rate above --max-error-rate (fraction, default 0.001 = 0.1%)
  * achieved throughput below --min-rps-ratio x --target-rps (default 0.9)
Exits 2 if the results are missing or unusable. Thresholds can also come from
the PERF_MAX_P99_MS, PERF_MAX_ERROR_RATE, PERF_MIN_RPS_RATIO, TARGET_RPS env vars.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Thresholds:
    target_rps: float
    max_p99_ms: float = 250.0
    max_error_rate: float = 0.001
    min_rps_ratio: float = 0.9


@dataclass(frozen=True)
class Summary:
    requests: int
    failures: int
    p99_ms: float
    rps: float

    @property
    def error_rate(self) -> float:
        return self.failures / self.requests if self.requests else 1.0


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    actual: str
    limit: str


class ResultsError(Exception):
    pass


def read_aggregated(path: Path) -> Summary:
    if not path.exists():
        raise ResultsError(f"{path} not found")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    agg = next((r for r in rows if r.get("Name") == "Aggregated"), None)
    if agg is None:
        raise ResultsError(f"no Aggregated row in {path}")
    try:
        summary = Summary(
            requests=int(agg["Request Count"]),
            failures=int(agg["Failure Count"]),
            p99_ms=float(agg["99%"]),
            rps=float(agg["Requests/s"]),
        )
    except (KeyError, ValueError) as exc:
        raise ResultsError(f"unreadable Aggregated row in {path}: {exc}") from exc
    if summary.requests == 0:
        raise ResultsError("no requests were recorded after warmup")
    return summary


def evaluate(summary: Summary, t: Thresholds) -> list[Check]:
    min_rps = t.target_rps * t.min_rps_ratio
    return [
        Check("p99 latency", summary.p99_ms <= t.max_p99_ms, f"{summary.p99_ms:.0f} ms", f"<= {t.max_p99_ms:.0f} ms"),
        Check(
            "error rate",
            summary.error_rate <= t.max_error_rate,
            f"{summary.error_rate:.4%} ({summary.failures}/{summary.requests})",
            f"<= {t.max_error_rate:.2%}",
        ),
        Check("throughput", summary.rps >= min_rps, f"{summary.rps:.1f} rps", f">= {min_rps:.1f} rps"),
    ]


def report(checks: Sequence[Check]) -> str:
    lines = ["| check | result | actual | limit |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.actual} | {c.limit} |")
    return "\n".join(lines)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stats_csv", type=Path, help="Locust *_stats.csv")
    parser.add_argument("--target-rps", type=float, default=_env_float("TARGET_RPS", 50))
    parser.add_argument("--max-p99-ms", type=float, default=_env_float("PERF_MAX_P99_MS", 250))
    parser.add_argument("--max-error-rate", type=float, default=_env_float("PERF_MAX_ERROR_RATE", 0.001))
    parser.add_argument("--min-rps-ratio", type=float, default=_env_float("PERF_MIN_RPS_RATIO", 0.9))
    args = parser.parse_args(argv)

    thresholds = Thresholds(args.target_rps, args.max_p99_ms, args.max_error_rate, args.min_rps_ratio)
    try:
        summary = read_aggregated(args.stats_csv)
    except ResultsError as exc:
        print(f"performance gate: cannot evaluate: {exc}", file=sys.stderr)
        return 2

    checks = evaluate(summary, thresholds)
    table = report(checks)
    passed = all(c.passed for c in checks)
    verdict = "PASSED" if passed else "FAILED"
    print(f"performance gate {verdict}\n{table}")
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a") as f:
            f.write(f"### Performance gate {verdict}\n\n{table}\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
