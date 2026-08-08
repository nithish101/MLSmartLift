import threading

from app.cache import CacheStats, feature_hash, features_key, prediction_key
from app.features import ModelInput
from app.inference import HistoryFeatures
from app.features import TrainingContext


def test_feature_hash_ignores_key_order():
    assert feature_hash({"a": 1, "b": 2.0}) == feature_hash({"b": 2.0, "a": 1})
    assert feature_hash({"a": 1}) != feature_hash({"a": 2})


def test_equivalent_inputs_share_a_key():
    a = ModelInput(exercise="squat", weight=225, rir=2, bodyweight=180)
    b = ModelInput(exercise="squat", weight=225.0, rir=2, bodyweight=180.0, num_sets=4)
    assert prediction_key(1, a.as_row()) == prediction_key(1, b.as_row())


def test_failure_rir_and_zero_rir_share_a_key():
    a = ModelInput(exercise="squat", weight=225, rir=-1, bodyweight=180)
    b = ModelInput(exercise="squat", weight=225, rir=0, bodyweight=180)
    assert prediction_key(1, a.as_row()) == prediction_key(1, b.as_row())


def test_key_includes_model_version():
    row = ModelInput(exercise="squat", weight=225, rir=2, bodyweight=180).as_row()
    assert prediction_key(1, row) != prediction_key(2, row)
    assert prediction_key(3, row).startswith("smartlift:pred:v3:")
    assert features_key("deadlift") == "smartlift:features:deadlift"


def test_stats_snapshot_and_ratio():
    stats = CacheStats()
    for hit in (True, True, True, False):
        stats.record("prediction", hit)
    stats.record("features", False)
    assert stats.snapshot() == {
        "features": {"hits": 0, "misses": 1, "hit_ratio": 0.0},
        "prediction": {"hits": 3, "misses": 1, "hit_ratio": 0.75},
    }


def test_stats_thread_safe():
    stats = CacheStats()
    threads = [threading.Thread(target=lambda: [stats.record("p", True) for _ in range(1000)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert stats.snapshot()["p"]["hits"] == 8000


def test_history_features_round_trip():
    ctx = TrainingContext(
        last_weight=145, last_reps=6, last_rir=1, num_sets=4, frequency=3, volume_score=40, progression_trend=0.7
    )
    hf = HistoryFeatures(context=ctx, top_weights=[135, 140, 145], top_rirs=[2, 2, 1])
    assert HistoryFeatures.from_json(hf.to_json()) == hf
    empty = HistoryFeatures(context=None, top_weights=[], top_rirs=[])
    assert HistoryFeatures.from_json(empty.to_json()) == empty
