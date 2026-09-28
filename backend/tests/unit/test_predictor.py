from collections.abc import Sequence

import pytest

from app.features import ModelInput, TrainingContext
from app.ml import predictor


class FixedCurveModel:
    """Predicts reps as a linear function of load: 20 - weight / 15."""

    def __init__(self) -> None:
        self.calls: list[ModelInput] = []

    def predict(self, inputs: Sequence[ModelInput]) -> list[float]:
        self.calls.extend(inputs)
        return [20 - i.weight / 15 for i in inputs]


def ctx(weight: float) -> TrainingContext:
    return TrainingContext(
        last_weight=weight, last_reps=8, last_rir=2, num_sets=4, frequency=3, volume_score=40, progression_trend=0.8
    )


@pytest.mark.parametrize(
    ("reps", "expected"),
    [(1, "decrease"), (4, "decrease"), (5, "maintain"), (9, "maintain"), (10, "increase")],
)
def test_outcome_buckets(reps, expected):
    assert predictor.outcome_for(reps) == expected


def test_round_reps_floor_of_one():
    assert predictor.round_reps(-3.2) == 1
    assert predictor.round_reps(6.6) == 7


class TestRecommendation:
    def test_maintains_weight_inside_window(self):
        # 150 lbs -> 10 reps? 20 - 10 = 10 -> increase; use 180 -> 8 reps.
        rec = predictor.recommend_next_session(FixedCurveModel(), "squat", 180, ctx(180))
        assert rec["recommended_weight"] == 180
        assert rec["expected_reps"] == 8
        assert rec["target_rir"] == 2

    def test_increases_when_too_easy(self):
        model = FixedCurveModel()
        rec = predictor.recommend_next_session(model, "squat", 180, ctx(120))  # 12 reps
        assert rec["recommended_weight"] == 125
        assert len(model.calls) == 2
        assert model.calls[1].weight == 125

    def test_decreases_when_too_hard(self):
        rec = predictor.recommend_next_session(FixedCurveModel(), "deadlift", 180, ctx(250))  # 3 reps
        assert rec["recommended_weight"] == 245

    def test_never_below_empty_bar(self):
        class TooHard:
            def predict(self, inputs):
                return [2.0 for _ in inputs]

        rec = predictor.recommend_next_session(TooHard(), "bench_press", 180, ctx(45))
        assert rec["recommended_weight"] == 45

    def test_passes_history_features_to_model(self):
        model = FixedCurveModel()
        predictor.recommend_next_session(model, "bench_press", 175, ctx(180))
        sent = model.calls[0]
        assert (sent.num_sets, sent.frequency, sent.volume_score, sent.progression_trend) == (4, 3, 40, 0.8)
        assert sent.bodyweight == 175
        assert sent.rir == predictor.TARGET_RIR

    def test_confidence_peaks_mid_window(self):
        mid = predictor.recommend_next_session(FixedCurveModel(), "squat", 180, ctx(195))  # 7 reps
        edge = predictor.recommend_next_session(FixedCurveModel(), "squat", 180, ctx(165))  # 9 reps
        assert mid["confidence"] == 1.0
        assert edge["confidence"] < mid["confidence"]


class TestOneRepMax:
    def test_epley_with_rir(self):
        assert predictor.estimate_1rm(200, 5, 0) == pytest.approx(233.3, abs=0.1)
        assert predictor.estimate_1rm(200, 3, 2) == predictor.estimate_1rm(200, 5, 0)

    def test_single_to_failure_is_the_weight(self):
        assert predictor.estimate_1rm(315, 1, 0) == 315

    def test_negative_rir_treated_as_failure(self):
        assert predictor.estimate_1rm(200, 5, -1) == predictor.estimate_1rm(200, 5, 0)


class TestRecovery:
    def test_harder_sessions_need_more_recovery(self):
        easy = predictor.estimate_recovery_time([4, 4, 4], "bench_press")
        hard = predictor.estimate_recovery_time([0, 0, 0, 0, 0], "bench_press")
        assert hard["recovery_hours"] > easy["recovery_hours"]

    def test_clamped_and_formatted(self):
        result = predictor.estimate_recovery_time([0] * 20, "deadlift")
        assert result == {"recovery_hours": 72.0, "recovery_estimate": "3 days"}
        result = predictor.estimate_recovery_time([2, 2], "squat")
        assert result["recovery_hours"] == pytest.approx(52.8)
        assert result["recovery_estimate"] == "2 days 4 hours"

    def test_unknown_exercise_uses_default_base(self):
        assert predictor.estimate_recovery_time([], "curl")["recovery_hours"] == 40.0


class TestAlerts:
    def test_no_alerts_without_history(self):
        assert predictor.generate_alerts([135], [2], 50) == []

    def test_regression_warning(self):
        alerts = predictor.generate_alerts([150, 145, 140], [2, 2, 2], 50)
        assert [a["type"] for a in alerts] == ["warning"]

    def test_plateau_info(self):
        alerts = predictor.generate_alerts([150, 150, 150, 150], [2, 2, 2, 2], 50)
        assert "plateaued" in alerts[0]["message"]

    def test_volume_alerts(self):
        assert any("High training stress" in a["message"] for a in predictor.generate_alerts([1, 2], [2, 2], 90))
        assert any("too low" in a["message"] for a in predictor.generate_alerts([1, 2], [2, 2], 10))

    def test_high_rir_tip(self):
        alerts = predictor.generate_alerts([100, 105], [4, 4], 50)
        assert alerts == [
            {"type": "tip", "message": "Average RIR is high. You could increase intensity for better stimulus."}
        ]
