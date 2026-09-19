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

## The ML side

The model is a `GradientBoostingRegressor` from scikit-learn. It takes in:

| Feature | Importance |
|---------|-----------|
| Weight lifted | 35.7% |
| Bodyweight | 32.7% |
| Progression trend | 11.8% |
| Exercise type | 10.2% |
| Volume score | 5.5% |
| Sets | 1.6% |
| RIR | 1.5% |
| Frequency | 1.1% |

Frequency, sets, volume score and progression trend are derived from your logged history (sessions in the last week, average sets per session, the latest session's training stress, and the slope of your top-set weight over the last five sessions).

It predicts how many reps you'll get at a given weight. From there, the recommendation engine decides whether to increase weight (if predicted reps > 9), decrease (if < 5), or stay put. Target range is 5–9 reps at RIR 2, which is a solid hypertrophy zone.

Current performance: **MAE of 1.15 reps** (RMSE 1.41) on a held-out 20% of the synthetic data. Not perfect, but good enough to be useful.

1RM is estimated with the Epley formula (adjusted for RIR):

```
e1RM = weight × (1 + (reps + RIR) / 30)
```

## Tech stack

- **Backend**: FastAPI + PostgreSQL (SQLAlchemy) + scikit-learn
- **Model registry**: MLflow (tracking server + registry; the API serves whatever version is in `Production`)
- **Cache**: Redis — inference results keyed by model version + a SHA-256 of the feature vector (TTL 1h), and the per-exercise history features the recommendation endpoints derive from Postgres (TTL 5 min, invalidated when a workout is logged). If Redis is down the API keeps serving uncached. Hit/miss counts are reported in `GET /health`.
- **Observability**: Prometheus metrics at `GET /metrics` (request latency by route, inference latency split by cache hit/miss, cache hit ratio, the serving model version, predictions by outcome), JSON logs with a per-request `X-Request-ID`, and a Grafana dashboard provisioned from `observability/grafana/dashboards/smartlift.json`.
- **Drift detection**: each training run logs the training feature distribution next to the model in MLflow. Every served prediction's features are written (buffered, off the request path) to `inference_log`, and the `drift-worker` container runs an APScheduler job every 5 minutes that computes the Population Stability Index of each feature over the last 24h of traffic for the Production version. Results go to the `drift_results` table and `GET /drift`, PSI is exported as `smartlift_feature_psi{feature}`, and any feature above 0.2 is logged as a warning.
- **Frontend**: React (Vite) with Recharts for visualizations
- **Styling**: Custom CSS, dark theme, mobile-first
- **Deployment**: PWA-ready (can be added to iPhone home screen)

## Running it locally

### With Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
# API      → http://localhost:8000 (docs at /docs)
# Frontend → http://localhost:5173
# MLflow   → http://localhost:5050
# Grafana  → http://localhost:3000 (dashboard "SmartLift API")
# Prometheus → http://localhost:9090

# Optional: load ~4 weeks of demo workouts
docker compose exec api python -m app.seed
```

On a fresh start the one-shot `trainer` service trains a model, registers it in MLflow and promotes it to `Production`; the API waits for it and then loads the model from the registry.

### Training and promoting a model

```bash
# Train a new version and park it in Staging
docker compose run --rm trainer python -m app.ml.train --stage Staging

# See what's registered, then promote
docker compose exec api python -m app.registry list
docker compose exec api python -m app.registry promote --version 2 --stage Production
```

The API checks the registry every `MODEL_REFRESH_SECONDS` (default 30) and hot-swaps to the new Production version — no restart or redeploy. `GET /version` shows the version currently serving.

### Backend without Docker

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/), and a PostgreSQL and MLflow server to point at (`docker compose up postgres mlflow` works).

```bash
cd backend
uv sync
export DATABASE_URL=postgresql+psycopg://smartlift:smartlift@localhost:5432/smartlift
export MLFLOW_TRACKING_URI=http://localhost:5050   # e.g. the compose mlflow service
uv run python -m app.ml.train --bootstrap
uv run uvicorn app.main:app --reload
```

### Frontend without Docker

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### iPhone access

Start the frontend with `npm run dev -- --host`, then open your local IP in Safari and add to home screen. It'll run as a standalone app without the browser chrome.

## Tests

```bash
cd backend
uv run pytest
```

Unit tests cover feature computation, input validation and the recommendation logic. Integration tests start throwaway PostgreSQL, Redis and MLflow containers with [testcontainers](https://testcontainers.com/) (Docker must be running) and hit the real API, including `POST /predict` against a model loaded from the registry, a promotion that the running app picks up without a restart, and the cold → warm cache path. Coverage is reported on `app/` and the run fails below 70%. To reuse running services instead of containers, set `TEST_DATABASE_URL` / `TEST_REDIS_URL` / `TEST_MLFLOW_URI`.

## Load testing

```bash
backend/loadtest/run.sh   # starts the stack, seeds it, runs Locust, applies the gate
```

Locust runs in its own container on the compose network and drives `POST /predict` (90%) and `GET /recommendation/{exercise}` (10%) at `TARGET_RPS` (default 50) using feature vectors from the same synthetic generator as training. The first `WARMUP_SECONDS` (30) ramp up and are then discarded; the following `TEST_SECONDS` (120) are measured. `backend/loadtest/check_results.py` then fails (exit 1) if p99 latency exceeds `PERF_MAX_P99_MS` (250), the error rate exceeds 0.1%, or achieved throughput is below 90% of the target. CSV results land in `backend/loadtest/results/`.

## Project structure

```
docker-compose.yml   — postgres, redis, mlflow, trainer (bootstrap), api, drift-worker, prometheus, grafana, frontend
                       (+ loadtest, behind the "loadtest" profile)
observability/       — Prometheus scrape config, Grafana provisioning + dashboard JSON
backend/
  app/
    main.py          — FastAPI app and endpoints
    db.py            — SQLAlchemy engine and tables
    schemas.py       — request/response schemas
    features.py      — feature validation and history-derived features
    repository.py    — database queries
    cache.py         — Redis cache with hit/miss accounting
    inference.py     — cached predictions and cached history-feature lookups
    metrics.py       — Prometheus metric definitions
    observability.py — JSON logging, request-id + latency middleware
    inference_log.py — buffered writer for served-prediction features
    drift.py         — training reference, PSI, drift check
    drift_worker.py  — APScheduler process running the drift check
    model_service.py — loads the Production model from MLflow, hot-swaps on promotion
    registry.py      — stage lookup/promotion helpers + `python -m app.registry` CLI
    seed.py          — populates the DB with demo workouts
    ml/
      generate_data.py — synthetic training data
      train.py         — training, MLflow logging and registration
      predictor.py     — recommendations, 1RM, recovery, alerts
  tests/
    unit/            — pure logic
    integration/     — API against real Postgres, Redis and MLflow
  loadtest/
    locustfile.py    — load profile with warmup
    check_results.py — performance gate over Locust's CSV
    run.sh           — run against the compose stack

frontend/
  src/
    App.jsx          — tab navigation shell
    api.js           — API client
    components/
      WorkoutLogger.jsx   — log exercises and sets
      Dashboard.jsx       — charts and progress tracking
      Recommendation.jsx  — AI predictions and alerts
      ModelExplainer.jsx  — how the model works (for demo)
```

## What I'd do next

- Expand the exercise library beyond the big 3
- Train on real anonymized data instead of synthetic
- Add adaptive retraining so the model improves per user
- Build out a full training program generator
- Injury risk estimation based on volume/intensity patterns
