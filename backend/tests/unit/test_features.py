from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.features import (
    DEFAULT_PROGRESSION_TREND,
    FEATURE_COLUMNS,
    ModelInput,
    SessionSummary,
    calculate_volume_score,
    compute_context,
    compute_frequency,
    compute_progression_trend,
    to_frame,
)

T0 = datetime(2026, 7, 1, 18, 0, tzinfo=UTC)


def session(days: float, weight: float, reps: int = 8, rir: int = 2, n_sets: int = 4) -> SessionSummary:
    return SessionSummary(
        performed_at=T0 + timedelta(days=days),
        top_weight=weight,
        top_reps=reps,
        top_rir=rir,
        set_rirs=tuple([rir] * n_sets),
    )


VALID = {"exercise": "squat", "weight": 225, "rir": 2, "bodyweight": 180}


class TestModelInput:
    def test_defaults_fill_history_features(self):
        m = ModelInput(**VALID)
        assert (m.num_sets, m.frequency, m.volume_score, m.progression_trend) == (4, 3, 50, 0.5)

    def test_failure_rir_maps_to_zero(self):
        assert ModelInput(**{**VALID, "rir": -1}).rir == 0

    @pytest.mark.parametrize(
        "field,value",
        [
            ("exercise", "curl"),
            ("weight", 0),
            ("weight", -45),
            ("weight", 1501),
            ("weight", float("nan")),
            ("weight", float("inf")),
            ("rir", 5),
            ("rir", -2),
            ("bodyweight", 20),
            ("num_sets", 0),
            ("frequency", 15),
            ("volume_score", 101),
            ("progression_trend", -0.1),
            ("progression_trend", 2.0),
        ],
    )
    def test_rejects_out_of_range(self, field, value):
        with pytest.raises(ValidationError):
            ModelInput(**{**VALID, field: value})

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            ModelInput(**VALID, sneaky=1)

    def test_rejects_missing_required(self):
        with pytest.raises(ValidationError):
            ModelInput(exercise="squat", weight=100, rir=2)

    def test_frame_has_training_column_order(self):
        df = to_frame([ModelInput(**VALID), ModelInput(**{**VALID, "exercise": "deadlift"})])
        assert list(df.columns) == FEATURE_COLUMNS
        assert df["exercise"].tolist() == ["squat", "deadlift"]
        assert len(df) == 2


class TestVolumeScore:
    def test_scales_with_intensity_frequency_and_muscle(self):
        assert calculate_volume_score([2, 2, 2, 2], 3, "bench_press") == int(4 * 0.6 * 3 * 1.0 * 8)
        assert calculate_volume_score([2, 2, 2, 2], 3, "deadlift") > calculate_volume_score(
            [2, 2, 2, 2], 3, "bench_press"
        )

    def test_clipped_to_0_100(self):
        assert calculate_volume_score([0] * 20, 7, "deadlift") == 100
        assert calculate_volume_score([], 3, "squat") == 0

    def test_easy_sets_contribute_nothing(self):
        assert calculate_volume_score([5, 5], 3, "squat") == 0


class TestHistoryFeatures:
    def test_frequency_counts_sessions_in_trailing_week(self):
        sessions = [session(0, 100), session(5, 100), session(8, 100), session(10, 100), session(12, 100)]
        # window is (day 5, day 12]: days 8, 10, 12
        assert compute_frequency(sessions) == 3

    def test_frequency_is_at_least_one(self):
        assert compute_frequency([session(0, 100)]) == 1

    def test_trend_uses_slope_in_5lb_units(self):
        assert compute_progression_trend([100, 105, 110, 115]) == pytest.approx(1.0)
        assert compute_progression_trend([100, 102.5, 105]) == pytest.approx(0.5)

    def test_trend_clipped_to_training_range(self):
        assert compute_progression_trend([100, 95, 90]) == 0.0
        assert compute_progression_trend([100, 120, 140]) == 1.5

    def test_trend_only_looks_at_last_five(self):
        assert compute_progression_trend([300, 100, 100, 100, 100, 100]) == 0.0

    def test_trend_default_without_enough_history(self):
        assert compute_progression_trend([135]) == DEFAULT_PROGRESSION_TREND
        assert compute_progression_trend([]) == DEFAULT_PROGRESSION_TREND

    def test_context_from_history(self):
        sessions = [
            session(0, 135, n_sets=3),
            session(3, 140, n_sets=4),
            session(5, 145, reps=6, rir=1, n_sets=5),
        ]
        ctx = compute_context(sessions, "bench_press")
        assert ctx is not None
        assert ctx.last_weight == 145
        assert ctx.last_reps == 6
        assert ctx.last_rir == 1
        assert ctx.num_sets == 4
        assert ctx.frequency == 3
        assert ctx.volume_score == calculate_volume_score([1] * 5, 3, "bench_press")
        assert ctx.progression_trend == pytest.approx(1.0)

    def test_context_empty_history(self):
        assert compute_context([], "squat") is None
