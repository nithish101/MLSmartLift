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
        to_frame(
            [
                ModelInput(exercise="squat", weight=135, rir=2, bodyweight=180),
                ModelInput(exercise="squat", weight=365, rir=2, bodyweight=180),
            ]
        )
    )
    assert heavy < light


def test_pipeline_needs_only_the_reviewed_skops_types(trained):
    import skops.io

    from app.ml.train import SKOPS_TRUSTED_TYPES

    untrusted = skops.io.get_untrusted_types(data=skops.io.dumps(trained.pipeline))
    assert set(untrusted) <= set(SKOPS_TRUSTED_TYPES)


def test_fast_path_matches_pipeline_exactly(trained, training_frame):
    from app.model_service import LoadedModel, fast_path
    from loadtest.traffic import build_pool

    fast = fast_path(trained.pipeline)
    assert fast is not None
    inputs = [ModelInput(**row) for row in build_pool(num_lifters=40)]
    via_pipeline = trained.pipeline.predict(to_frame(inputs))
    assert np.array_equal(fast(inputs), via_pipeline)  # bit-for-bit, not approximately
    loaded = LoadedModel(ref=None, model=trained.pipeline, loaded_at=None, fast=fast)  # type: ignore[arg-type]
    assert loaded.predict(inputs[:3]) == [float(v) for v in via_pipeline[:3]]


def test_fast_path_declines_unfamiliar_models(trained):
    from sklearn.dummy import DummyRegressor
    from sklearn.pipeline import Pipeline

    from app.model_service import fast_path

    assert fast_path(DummyRegressor()) is None
    assert fast_path(Pipeline([("model", DummyRegressor())])) is None
    reordered = Pipeline([("features", trained.pipeline.named_steps["features"]), ("other", DummyRegressor())])
    assert fast_path(reordered) is None

    # A numeric step that actually transforms must not be bypassed.
    import copy

    import numpy as np
    from sklearn.preprocessing import FunctionTransformer

    altered = copy.deepcopy(trained.pipeline)
    ct = altered.named_steps["features"]
    ct.transformers_ = [(n, FunctionTransformer(np.log1p) if n == "numeric" else t, c) for n, t, c in ct.transformers_]
    assert fast_path(altered) is None
