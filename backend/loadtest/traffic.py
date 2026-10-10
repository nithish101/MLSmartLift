"""
Request bodies for the load test.

Feature vectors come from the same synthetic generator as the training data,
but from different simulated lifters (seed 2026). The pool is large enough that
its feature distributions match training (PSI well under the 0.2 drift
threshold, see tests/unit/test_perf_gate.py), so load tests exercise the API
like real traffic without tripping the drift detector. Vectors repeat across
users, which gives the cache a realistic hit rate.
"""

from __future__ import annotations

from typing import Any

from app.features import FEATURE_COLUMNS
from app.ml.generate_data import generate_dataset

POOL_LIFTERS = 200
POOL_SEED = 2026


def build_pool(num_lifters: int = POOL_LIFTERS, seed: int = POOL_SEED) -> list[dict[str, Any]]:
    frame = generate_dataset(num_lifters=num_lifters, seed=seed)[FEATURE_COLUMNS]
    return [
        {str(k): (v.item() if hasattr(v, "item") else v) for k, v in row.items()} for row in frame.to_dict("records")
    ]
