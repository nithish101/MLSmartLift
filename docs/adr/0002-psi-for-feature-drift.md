# 0002. Detect feature drift with the Population Stability Index

- Status: Accepted
- Decided: 2026-09-06 (Phase 5)

## Context

The model is trained on synthetic progressions. If real lifters look different (heavier, stronger, training more often), predictions degrade silently. Ground truth (actual reps) only arrives for logged workouts, so we need a label-free signal that the inputs no longer look like the training data.

The options we considered:

- **Kolmogorov–Smirnov / chi-squared tests.** These answer "is there any difference?", and with thousands of samples they flag trivial shifts as significant. They also need a different test for continuous and categorical features.
- **Jensen–Shannon / KL divergence.** These are close relatives of PSI, but have no widely shared thresholds.
- **Model-based drift (a domain classifier).** More sensitive, but it needs training and tuning of its own and is harder to explain.
- **PSI.** It works the same way on binned continuous and categorical features, it is cheap to compute from bin counts, it is symmetric, and it has conventional thresholds that are widely used for scorecard monitoring: below 0.1 is stable, 0.1 to 0.2 is a moderate shift, above 0.2 is significant.

## Decision

- At training time, compute a reference for every model feature: deciles of the training data for continuous features, and one bin per value (plus an "unseen" bin) for discrete features with 12 or fewer distinct values. Log it as `drift/reference.json` on the model's MLflow run, so the reference always belongs to the version being served.
- Record the features of every served prediction in `inference_log`, through a buffered writer that keeps Postgres off the request path.
- A `drift-worker` process (APScheduler) computes PSI per feature every 5 minutes over the last 24 hours of traffic for the Production version, once at least 200 predictions are in the window. Empty bins are floored at 1e-4 so the log term stays finite.
- Results are written to `drift_results`, exposed via `GET /drift` and the `smartlift_feature_psi{feature}` gauge, and every feature above **0.2** is logged as a warning.

## Consequences

- PSI says that a feature moved, not whether accuracy suffered. It is a trigger to look (and maybe retrain), not an automatic action.
- Decile bins come from the training data, so shifts within a bin are invisible. That is acceptable for a coarse alarm.
- The window minimum (200) avoids noisy PSI from tiny samples, but means low-traffic periods report `insufficient_data` instead of a score.
- Repeated identical requests (cache hits) are still logged, because they are still traffic the model is serving.
- `inference_log` grows with traffic. Retention or partitioning will be needed before real-world volumes.
