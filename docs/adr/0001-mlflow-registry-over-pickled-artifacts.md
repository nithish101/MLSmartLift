# 0001. Serve models from the MLflow registry, not pickles on disk

- Status: Accepted
- Decided: 2026-07-26 (Phase 2)

## Context

The original backend loaded `smartlift_model.pkl` and `exercise_encoder.pkl` from the working directory at import time. That meant:

- No record of which data, hyperparameters or metrics produced the file being served. Retraining overwrote it in place.
- Changing the model meant rebuilding or redeploying the API, and rolling back meant finding an old file.
- `pickle` executes arbitrary code on load, so anyone who can write the file can run code in the API.
- The encoder and the model were separate files that had to stay in sync by convention.

## Decision

- Training logs every run to an MLflow tracking server (params, MAE/RMSE/R², feature importances, the drift reference) and registers the fitted model as a new version of `smartlift-reps`.
- The model is a single scikit-learn `Pipeline` (ordinal encoder + gradient boosting), so preprocessing is versioned with the estimator.
- The API resolves the version currently in a registry **stage** (`MODEL_STAGE`, default `Production`) at startup, then polls every `MODEL_REFRESH_SECONDS` and hot-swaps when a different version is promoted. `python -m app.registry promote` moves versions between `Staging` and `Production` and archives the previous holder, so a promotion or rollback needs no redeploy. `GET /version` and the `smartlift_model_version` gauge show what's serving.
- Models are serialized with skops rather than pickle. Loading only allows reviewed types (`SKOPS_TRUSTED_TYPES`), and a test asserts that the pipeline needs nothing else.
- Cached predictions are keyed by model version, so a promotion can never serve stale results.

## Consequences

- One more service (MLflow, SQLite backend plus a proxied artifact store) in the stack, and a one-shot `trainer` service to bootstrap the first Production model.
- MLflow 3 has deprecated registry stages in favor of aliases. We use stages because the promotion workflow is stage-based, isolate the calls in `app/registry.py`, and mirror each stage to an alias (`production`, `staging`) on promotion, removing the alias of the stage a version leaves so aliases never disagree with stages. Moving to aliases later means changing `latest_in_stage` to read the alias.
- If the registry is unreachable at startup the API starts degraded (`/predict` returns 503) and keeps retrying. Once a model is loaded, registry outages don't affect serving. MLflow's HTTP retries are capped so an outage can't stall startup for minutes.
- Promotions take effect within one poll interval (30 s by default), not instantly.
