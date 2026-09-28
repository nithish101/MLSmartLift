import math

import numpy as np
import pytest

from app.drift import bin_proportions, compute_reference, feature_psi, psi
from app.features import FEATURE_COLUMNS


class TestPsi:
    def test_identical_distributions_score_zero(self):
        assert psi([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]) == 0

    def test_matches_hand_computation(self):
        expected, actual = [0.5, 0.5], [0.3, 0.7]
        hand = (0.3 - 0.5) * math.log(0.3 / 0.5) + (0.7 - 0.5) * math.log(0.7 / 0.5)
        assert psi(expected, actual) == pytest.approx(hand)

    def test_is_symmetric_and_non_negative(self):
        a, b = [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]
        assert psi(a, b) == pytest.approx(psi(b, a))
        assert psi(a, b) > 0

    def test_empty_bins_stay_finite(self):
        assert math.isfinite(psi([0.5, 0.5, 0.0], [0.0, 0.5, 0.5]))

    def test_bin_count_mismatch(self):
        with pytest.raises(ValueError, match="same number of bins"):
            psi([1.0], [0.5, 0.5])


class TestReference:
    @pytest.fixture(scope="class")
    def reference(self, training_frame):
        return compute_reference(training_frame)

    def test_covers_every_model_feature(self, reference, training_frame):
        assert set(reference["features"]) == set(FEATURE_COLUMNS)
        assert reference["n_samples"] == len(training_frame)

    def test_discrete_features_are_categorical(self, reference):
        kinds = {f: spec["kind"] for f, spec in reference["features"].items()}
        for f in ("exercise", "rir", "num_sets", "frequency"):
            assert kinds[f] == "categorical", f
        for f in ("weight", "bodyweight", "progression_trend", "volume_score"):
            assert kinds[f] == "numeric", f

    def test_proportions_sum_to_one(self, reference):
        for spec in reference["features"].values():
            assert sum(spec["proportions"]) == pytest.approx(1.0)

    def test_numeric_bins_are_roughly_deciles(self, reference):
        props = reference["features"]["bodyweight"]["proportions"]
        assert len(props) == 10
        assert max(props) < 0.15

    def test_categorical_reserves_unseen_bin(self, reference):
        spec = reference["features"]["num_sets"]
        assert len(spec["proportions"]) == len(spec["categories"]) + 1
        assert spec["proportions"][-1] == 0


class TestBinning:
    def test_numeric_outliers_fall_in_open_end_bins(self):
        spec = {"kind": "numeric", "edges": [10.0, 20.0]}
        assert bin_proportions([-100, 15, 15, 1e9], spec) == [0.25, 0.5, 0.25]

    def test_unseen_categories_counted(self):
        spec = {"kind": "categorical", "categories": ["a", "b"]}
        assert bin_proportions(["a", "c", "c", "b"], spec) == [0.25, 0.25, 0.5]

    def test_empty_input(self):
        assert bin_proportions([], {"kind": "categorical", "categories": [1]}) == [0.0, 0.0]


class TestFeaturePsi:
    def test_sample_from_training_distribution_is_stable(self, training_frame):
        reference = compute_reference(training_frame)
        sample = training_frame.sample(n=2000, random_state=1, replace=True)
        scores = feature_psi(reference, sample)
        assert set(scores) == set(FEATURE_COLUMNS)
        assert max(scores.values()) < 0.1

    def test_shifted_feature_is_flagged_alone(self, training_frame):
        reference = compute_reference(training_frame)
        shifted = training_frame.sample(n=2000, random_state=2, replace=True).copy()
        shifted["weight"] = shifted["weight"] * 1.6
        shifted["num_sets"] = np.where(np.arange(len(shifted)) % 2 == 0, 8, shifted["num_sets"])  # unseen value
        scores = feature_psi(reference, shifted)
        assert scores["weight"] > 0.2
        assert scores["num_sets"] > 0.2
        assert scores["bodyweight"] < 0.1

    def test_tolerates_reference_without_unseen_bin(self):
        reference = {"features": {"rir": {"kind": "categorical", "categories": [1, 2], "proportions": [0.5, 0.5]}}}
        import pandas as pd

        assert feature_psi(reference, pd.DataFrame({"rir": [1, 2]})) == {"rir": 0.0}

    def test_ignores_features_missing_from_live_frame(self, training_frame):
        import pandas as pd

        reference = compute_reference(training_frame, columns=["weight", "rir"])
        assert set(feature_psi(reference, pd.DataFrame({"rir": [2, 2, 3]}))) == {"rir"}
