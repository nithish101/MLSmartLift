import numpy as np
import pytest

from app.features import FEATURE_COLUMNS, ModelInput, to_frame
from app.ml.generate_data import generate_dataset


def test_dataset_is_deterministic_and_well_formed():
    a = generate_dataset(num_lifters=3, seed=11)
    b = generate_dataset(num_lifters=3, seed=11)
    assert a.equals(b)
    assert set(FEATURE_COLUMNS) | {"reps"} <= set(a.columns)
    assert a["reps"].between(1, 15).all()
    assert a["rir"].between(0, 4).all()
    assert a["volume_score"].between(0, 100).all()
    assert set(a["exercise"]) == {"bench_press", "squat", "deadlift"}


def test_different_seeds_differ():
    assert not generate_dataset(num_lifters=2, seed=1).equals(generate_dataset(num_lifters=2, seed=2))


def test_trained_model_beats_mean_baseline(trained, training_frame):
    baseline_mae = float(np.mean(np.abs(training_frame["reps"] - training_frame["reps"].mean())))
    assert trained.metrics["mae"] < baseline_mae
    assert trained.metrics["rmse"] >= trained.metrics["mae"]
    assert set(trained.feature_importance) == set(FEATURE_COLUMNS)
    assert sum(trained.feature_importance.values()) == pytest.approx(1.0)


def test_pipeline_accepts_feature_frame(trained):
    inputs = [ModelInput(exercise=e, weight=185, rir=2, bodyweight=180) for e in ("bench_press", "squat", "deadlift")]
    preds = trained.pipeline.predict(to_frame(inputs))
    assert preds.shape == (3,)
    assert np.isfinite(preds).all()


def test_heavier_load_predicts_fewer_reps(trained):
    light, heavy = trained.pipeline.predict(
        to_frame([
            ModelInput(exercise="squat", weight=135, rir=2, bodyweight=180),
            ModelInput(exercise="squat", weight=365, rir=2, bodyweight=180),
        ])
    )
    assert heavy < light


def test_pipeline_needs_only_the_reviewed_skops_types(trained):
    import skops.io

    from app.ml.train import SKOPS_TRUSTED_TYPES

    untrusted = skops.io.get_untrusted_types(data=skops.io.dumps(trained.pipeline))
    assert set(untrusted) <= set(SKOPS_TRUSTED_TYPES)
