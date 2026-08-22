"""Prometheus metrics. Scraped from GET /metrics."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Buckets tuned for a fast JSON API: sub-millisecond cache hits up to multi-second outliers.
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5, 5.0)

REQUEST_LATENCY = Histogram(
    "smartlift_http_request_duration_seconds",
    "HTTP request latency by route template",
    ["method", "route", "status"],
    buckets=LATENCY_BUCKETS,
)

INFERENCE_LATENCY = Histogram(
    "smartlift_inference_duration_seconds",
    "Time to produce one prediction, including the cache lookup",
    ["cache"],  # hit | miss
    buckets=LATENCY_BUCKETS,
)

CACHE_REQUESTS = Counter(
    "smartlift_cache_requests_total",
    "Cache lookups by cache and result",
    ["cache", "result"],  # result: hit | miss
)

CACHE_HIT_RATIO = Gauge(
    "smartlift_cache_hit_ratio",
    "Cumulative cache hit ratio since process start",
    ["cache"],
)

CACHE_ERRORS = Counter("smartlift_cache_errors_total", "Redis errors (treated as misses)")

MODEL_VERSION = Gauge(
    "smartlift_model_version",
    "Registry version of the model currently serving (0 = none loaded)",
    ["model_name", "stage"],
)

MODEL_RELOADS = Counter("smartlift_model_reloads_total", "Times a new model version was hot-swapped in")

PREDICTIONS = Counter(
    "smartlift_predictions_total",
    "Predictions by the weight change they imply",
    ["outcome"],  # decrease | maintain | increase
)
