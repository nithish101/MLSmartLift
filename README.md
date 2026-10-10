# SmartLift

SmartLift is a workout tracking app that uses machine learning to help you figure out what weight to lift next. I built it because I got tired of guessing whether I should go up in weight or stay where I am — so I trained a model to answer that for me.

## What it does

You log your workouts (exercise, weight, reps, how many reps you had left in the tank), and SmartLift uses a Gradient Boosting model to:

- **Predict your next session** — what weight to use, how many reps to expect, and what intensity to target
- **Track your estimated 1RM** over time so you can see if you're actually getting stronger
- **Score your weekly training volume** (0–100) to flag if you're undertraining or overdoing it
- **Compare predictions vs reality** so you can see how well the model is actually working
- **Estimate recovery time** after each session based on intensity and muscle group

Right now it supports bench press, squat, and deadlift. The model was trained on synthetic progression data that follows realistic strength curves.

## Screenshots

<div style="display: flex; gap: 10px; overflow-x: auto;">
  <img src="frontend/screenshots/dashboard.webp" width="24%" alt="Dashboard Screenshot" />
  <img src="frontend/screenshots/log.webp" width="24%" alt="Log Workout Screenshot" />
  <img src="frontend/screenshots/ai_coach.webp" width="24%" alt="AI Coach Screenshot" />
  <img src="frontend/screenshots/explainer.webp" width="24%" alt="Model Explainer Screenshot" />
</div>

### Operations

| | |
|---|---|
| ![Grafana dashboard: request rate, latency percentiles, cache hit ratio, inference latency](docs/images/grafana.webp) | ![Grafana drift panels: PSI per feature against the 0.2 threshold](docs/images/grafana_drift.webp) |
| **Grafana — SmartLift API.** Serving model version, request rate, 5xx rate, p50/p95/p99 latency per route, inference latency split by cache hit and miss. | **Feature drift.** PSI of every model feature against the Production model's training reference, with the 0.1 and 0.2 thresholds. |
| ![MLflow model registry with four versions; version 4 holds the production alias](docs/images/mlflow_registry.webp) | ![OpenAPI docs listing the SmartLift endpoints](docs/images/api_docs.webp) |
| **MLflow registry.** Every training run registers a version; promotion moves the `production` alias (and stage) and the API hot-swaps to it. | **API.** Interactive OpenAPI docs at `/docs`. |

The screenshots come from a local `docker compose up` with demo data (`python -m app.seed`), a few promotions, and the load generator running at 25 RPS.

## Architecture

```mermaid
flowchart LR
    user([Browser / PWA]) --> fe[frontend<br/>React + nginx]
    fe -->|REST| api

    subgraph serving [Serving]
        api[api<br/>FastAPI + uvicorn]
        redis[(Redis<br/>prediction + feature cache)]
        pg[(PostgreSQL<br/>workouts, inference_log,<br/>drift_results)]
        api <-->|cache by model version<br/>+ feature hash| redis
        api -->|workouts, buffered<br/>inference log| pg
    end

    subgraph ml [Model lifecycle]
        trainer[trainer<br/>train + register]
        mlflow[(MLflow<br/>tracking + registry)]
        trainer -->|params, MAE/RMSE,<br/>model, drift reference| mlflow
    end

    api -->|load Production version,<br/>poll for promotions| mlflow
    migrate[migrate<br/>alembic upgrade head] --> pg

    subgraph monitoring [Monitoring]
        drift[drift-worker<br/>APScheduler PSI job]
        prom[(Prometheus)]
        graf[Grafana<br/>SmartLift API dashboard]
        drift -->|live features| pg
        drift -->|training reference| mlflow
        drift -->|PSI results| pg
        prom -->|scrape /metrics| api
        prom -->|scrape :9101| drift
        graf --> prom
    end
```

- **Inference.** `POST /predict` (and the recommendation endpoints) validate the feature vector, check Redis under `pred:v<model version>:<sha256 of features>`, and on a miss run the scikit-learn pipeline loaded from the MLflow registry. History-derived features (training frequency, sets, volume score, progression trend) come from Postgres and are cached per exercise until the next workout is logged.
- **Model lifecycle.** Training logs runs and registers versions in MLflow. The API serves whatever version is in the `Production` stage and hot-swaps when that changes ([ADR 0001](docs/adr/0001-mlflow-registry-over-pickled-artifacts.md)).
- **Monitoring.** Every served prediction's features are logged. A worker compares them with the training distribution using PSI ([ADR 0002](docs/adr/0002-psi-for-feature-drift.md)). Prometheus scrapes both processes, and Grafana shows latency, cache, model version, prediction mix and drift.
- **Schema.** Alembic migrations own the schema. The one-shot `migrate` service applies them before anything else starts.

## The ML side

The model is a scikit-learn `Pipeline`: an ordinal encoder for the exercise followed by a `GradientBoostingRegressor`. It takes in:

| Feature | Importance |
|---------|-----------|
| Weight lifted | 35.7% |
| Bodyweight | 32.7% |
| Progression trend | 11.8% |
| Exercise type | 10.2% |
| Volume score | 5.4% |
| Sets | 1.5% |
| RIR | 1.5% |
| Frequency | 1.1% |

Frequency, sets, volume score and progression trend are derived from your logged history (sessions in the last week, average sets per session, the latest session's training stress, and the slope of your top-set weight over the last five sessions).

It predicts how many reps you'll get at a given weight. From there, the recommendation engine decides whether to increase weight (if predicted reps > 9), decrease (if < 5), or stay put. Target range is 5–9 reps at RIR 2, which is a solid hypertrophy zone.

Current performance: **MAE of 1.15 reps** (RMSE 1.41) on a held-out 20% of the synthetic data. Not perfect, but good enough to be useful. Every training run logs these metrics to MLflow, so you can compare versions before promoting one.

1RM is estimated with the Epley formula (adjusted for RIR):

```
e1RM = weight × (1 + (reps + RIR) / 30)
```

## Running it locally

Everything runs with Docker Compose. You need Docker with Compose v2 and about 4 GB of free memory.

```bash
cp .env.example .env        # then change the passwords
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| API (OpenAPI docs at `/docs`) | http://localhost:8000 |
| MLflow | http://localhost:5050 |
| Grafana (dashboard "SmartLift API") | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

On a fresh start the services come up in this order:

1. `postgres`, `redis` and `mlflow` start and report healthy.
2. `migrate` applies the Alembic migrations.
3. `trainer` trains a model, registers it as version 1 and promotes it to `Production`. It does nothing if a Production model already exists.
4. `api` loads the Production model from the registry.
5. `drift-worker`, `prometheus` and `grafana` start.

To load about four weeks of demo workouts:

```bash
docker compose exec api python -m app.seed
```

`docker compose down -v` wipes everything, including the registry.

### iPhone access

Start the frontend with `npm run dev -- --host` (from `frontend/`), then open your local IP in Safari and add to home screen. It'll run as a standalone app without the browser chrome.

## Training and promoting a model

```bash
# Train a new version and put it in Staging
docker compose run --rm trainer python -m app.ml.train --stage Staging

# Compare it with what's serving (metrics are on each run in the MLflow UI)
docker compose exec api python -m app.registry list

# Promote it; the previous Production version is archived, and the
# production/staging aliases are moved to match
docker compose exec api python -m app.registry promote --version 2 --stage Production

# Confirm the API switched over (within MODEL_REFRESH_SECONDS, default 30 s)
curl localhost:8000/version
```

There is no restart or redeploy. To roll back, promote the older version again. `train` options: `--lifters N` and `--seed S` for the synthetic data, `--data file.csv` to train on your own export, and `--bootstrap` to train only if nothing is in Production.

## API

| Endpoint | |
|---|---|
| `POST /predict` | predicted reps for one feature vector; response includes `model_version` and `cached` |
| `POST /workouts`, `GET /workouts` | log and list workouts |
| `GET /recommendation/{exercise}` | next-session weight, reps and confidence |
| `GET /progress/{exercise}`, `/volume-score`, `/predicted-vs-actual/{exercise}`, `/alerts/{exercise}` | dashboard data |
| `GET /health` | database, model and cache status, plus cache hit/miss counts |
| `GET /version` | app version and the registry model version being served |
| `GET /drift` | per-feature PSI from the latest drift check |
| `GET /metrics` | Prometheus metrics |

Every response carries an `X-Request-ID` header (yours if you sent a valid one), and every log line for that request includes it. Logs are one JSON object per line.

## Observability

- **Metrics** (`GET /metrics`): request latency histogram by route template, inference latency by cache hit/miss, cache requests and hit ratio, the serving model version, model reloads, predictions by outcome (decrease / maintain / increase), and dropped inference-log rows. The drift worker exports `smartlift_feature_psi{feature}`, window sample counts and check results on `:9101`.
- **Dashboard**: `observability/grafana/dashboards/smartlift.json` is provisioned automatically. It shows request rate, p50/p95/p99 latency, 5xx rate, cache hit ratio, model version, prediction mix and PSI per feature against the 0.2 threshold.
- **Drift**: every 5 minutes, PSI over the last 24 h of traffic for the Production version, once at least 200 predictions are in the window. Results are written to `drift_results` and features above 0.2 are logged as warnings. See [ADR 0002](docs/adr/0002-psi-for-feature-drift.md).

## Development

### Backend without Docker

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). Start the backing services with Compose:

```bash
docker compose up -d postgres redis mlflow
cd backend
uv sync
export DATABASE_URL=postgresql+psycopg://smartlift:change-me@localhost:5432/smartlift  # match .env
export MLFLOW_TRACKING_URI=http://localhost:5050 REDIS_URL=redis://localhost:6379/0
uv run python -m app.migrate
uv run python -m app.ml.train --bootstrap
uv run uvicorn app.main:app --reload
```

### Frontend without Docker

```bash
cd frontend
npm install
npm run dev
```

### Checks

```bash
cd backend
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest            # needs Docker: starts Postgres, Redis and MLflow with testcontainers
```

Unit tests cover feature computation, input validation, recommendations, PSI, cache keys, logging and the performance gate. Integration tests run the real API against real Postgres, Redis and MLflow containers. They cover `/predict` end to end, a promotion picked up by a running app, the cold → warm cache path and invalidation, Redis being down, request-id propagation, metrics, the drift job and scheduler, and the migrations themselves. Coverage is measured on `app/` and the run fails below 70% (currently ~98%). Set `TEST_DATABASE_URL`, `TEST_REDIS_URL` or `TEST_MLFLOW_URI` to reuse running services instead of containers. With Colima, also export `DOCKER_HOST=unix://$HOME/.colima/default/docker.sock` and `TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE=/var/run/docker.sock`.

### Database migrations

```bash
cd backend
uv run alembic revision --autogenerate -m "describe the change"   # after editing app/db.py
uv run alembic upgrade head
uv run alembic check   # fails if models and migrations disagree
```

### Load testing

```bash
backend/loadtest/run.sh             # starts the stack, seeds it, runs Locust, applies the gate
KEEP_STACK=1 backend/loadtest/run.sh
```

Locust runs in its own container on the compose network (the `loadtest` profile) and drives `POST /predict` (90%) and `GET /recommendation/{exercise}` (10%) at `TARGET_RPS` (default 50), using feature vectors from the same generator as the training data. It ramps up for `WARMUP_SECONDS` (30), resets its statistics, and then measures `TEST_SECONDS` (120). `backend/loadtest/check_results.py` exits 1 if p99 latency exceeds `PERF_MAX_P99_MS` (250), the error rate exceeds 0.1%, or achieved throughput falls below 90% of the target. CSVs land in `backend/loadtest/results/`. [ADR 0003](docs/adr/0003-performance-gate-thresholds.md) explains the numbers.

## CI

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

| Stage | Enforces |
|---|---|
| Lint (ruff) | `ruff check` (pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify, pytest style) and `ruff format --check` |
| Type check (mypy) | `mypy --strict` over `app/` and `loadtest/` |
| Tests + coverage (pytest) | unit + integration tests against real Postgres, Redis and MLflow (testcontainers); fails under 70% coverage |
| Migrations (alembic) | upgrade an empty database to head, `alembic check` (models match migrations), downgrade to base and upgrade again, exactly one head |
| Docker build | the multi-stage runtime image builds, runs as uid 10001, contains no build tooling and imports the app; frontend image builds |
| Load test gate (locust) | `backend/loadtest/run.sh` against the containerized stack; fails on p99, error-rate or throughput breach |

Lint, types, tests and migrations run in parallel. The Docker build only runs if all four pass, and the load test gate only runs if the build passes. `main` is protected: all six checks must pass before a pull request can merge, and the rule applies to admins too.

## Configuration

All settings are environment variables. `.env.example` documents the ones you would normally change: database credentials, cache TTLs, model stage and refresh interval, drift window and threshold, performance-gate thresholds, and Grafana credentials. No secrets are committed.

## Decisions

Architecture decision records are in [`docs/adr`](docs/adr/README.md):

- [0001 — Serve models from the MLflow registry, not pickles on disk](docs/adr/0001-mlflow-registry-over-pickled-artifacts.md)
- [0002 — Detect feature drift with the Population Stability Index](docs/adr/0002-psi-for-feature-drift.md)
- [0003 — Performance gate thresholds](docs/adr/0003-performance-gate-thresholds.md)

## Project structure

```
docker-compose.yml     postgres, redis, mlflow, migrate, trainer, api, drift-worker,
                       prometheus, grafana, frontend (+ loadtest behind the "loadtest" profile)
.env.example           every tunable, no secrets
.github/workflows/     CI pipeline
docs/adr/              architecture decision records
observability/         Prometheus scrape config, Grafana provisioning + dashboard JSON
backend/
  Dockerfile           multi-stage: uv build stage -> slim non-root runtime
  alembic.ini
  app/
    main.py            FastAPI app and endpoints
    config.py          settings (environment variables)
    db.py              SQLAlchemy models
    migrations/        Alembic environment and versions
    migrate.py         applies migrations
    schemas.py         request/response schemas
    features.py        feature validation and history-derived features
    repository.py      database queries
    model_service.py   loads the Production model from MLflow, hot-swaps on promotion
    registry.py        stage lookup/promotion + `python -m app.registry`
    inference.py       cached predictions and cached feature lookups
    cache.py           Redis cache with hit/miss accounting
    inference_log.py   buffered writer for served-prediction features
    drift.py           training reference, PSI, drift check
    drift_worker.py    APScheduler process running the drift check
    metrics.py         Prometheus metric definitions
    observability.py   JSON logging, request-id + latency middleware
    seed.py            demo workouts
    ml/
      generate_data.py synthetic training data
      train.py         training, MLflow logging and registration
      predictor.py     recommendations, 1RM, recovery, alerts
  tests/
    unit/              pure logic
    integration/       against real Postgres, Redis and MLflow
  loadtest/
    locustfile.py      load profile with warmup
    check_results.py   performance gate
    run.sh             run against the compose stack
    Dockerfile         Locust image for the compose "loadtest" service
frontend/
  src/
    App.jsx            tab navigation shell
    api.js             API client
    components/        WorkoutLogger, Dashboard, Recommendation, ModelExplainer
```

## What I'd do next

- Expand the exercise library beyond the big 3
- Train on real anonymized data instead of synthetic
- Add adaptive retraining so the model improves per user
- Build out a full training program generator
- Injury risk estimation based on volume/intensity patterns
- Retention for `inference_log` once traffic is real
