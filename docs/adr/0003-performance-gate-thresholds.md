# 0003. Performance gate thresholds

- Status: Accepted
- Decided: 2026-09-19 (Phase 6)

## Context

CI runs a Locust load test against the containerized stack (`backend/loadtest/run.sh`) and fails the build if the API regresses. The gate has to catch real regressions, such as an uncached hot path, a blocking call on the event loop, an N+1 query, or a model that's ten times slower, without failing on the normal noise of a shared CI runner.

Measured profile: 90% `POST /predict`, 10% `GET /recommendation/{exercise}`, realistic feature vectors with natural repeats, single uvicorn worker, everything on one host. On a 4-vCPU laptop VM (Colima), steady-state runs at 50 RPS gave **p99 37–140 ms**, a median of 7–16 ms, and **0 errors** in 6,000+ requests per run. Serial latency inside the container is about 1 ms. Most of the tail comes from the virtualized network and scheduler, not the request path (the API sat at 6–12% CPU under this load). Cache misses add about 1 ms of model inference.

## Decision

| Setting | Value | Why |
|---|---|---|
| `TARGET_RPS` | 50 | Comfortably above what a personal training app sees, and enough concurrency (50 users) to expose contention, pool exhaustion or event-loop blocking. Low enough that a single runner's load generator isn't the bottleneck. |
| `WARMUP_SECONDS` | 30 | Covers connection-pool fill, the first model-inference calls and a cold Redis cache. Stats are reset afterwards so warmup doesn't count. |
| `TEST_SECONDS` | 120 | About 6,000 requests, enough that p99 rests on ~60 samples and a 0.1% error budget means at most 6 failures, while keeping the CI job short. |
| `PERF_MAX_P99_MS` | 250 | Roughly 2× the worst healthy p99 we measured. A real regression (a synchronous network call on the hot path, losing the cache, an O(n) query on history) shows up as hundreds of ms. Runner jitter doesn't get there. |
| `PERF_MAX_ERROR_RATE` | 0.1% | The endpoints are deterministic and should never fail under this load. 0.1% tolerates a single transient connection reset per ~1,000 requests, and anything systematic blows straight through it. |
| `PERF_MIN_RPS_RATIO` | 0.9 | Users pace themselves with `constant_throughput(1)`, so achieved RPS only falls short when responses are slow or the service stalls. 10% slack absorbs ramp edges and scheduler hiccups. |

The gate script exits 1 on any breach, and 2 if Locust produced no usable results, so a broken harness also fails the build.

## Consequences

- p99 measured on a shared runner is noisy. If the gate flaps without code changes, the first fix is running the generator on a dedicated runner, not loosening the threshold.
- Throughput is measured on Locust's wall clock, so pausing the host (a laptop sleeping, a VM being suspended) registers as a throughput breach. That is correct for CI, but local runs need a machine that stays awake.
- Thresholds live in `.env.example`, `docker-compose.yml` and the gate's flags. Change them there and update this record with new measurements.
- The gate guards the API's own serving path. It does not load-test MLflow, the drift worker or Grafana.
