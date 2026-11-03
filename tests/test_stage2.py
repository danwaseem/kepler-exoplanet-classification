"""Guards for the Stage 2 development-only Decision Tree."""

import numpy as np
import pandas as pd
import pytest

from exoplanet.stage2 import assert_primary_features, development_only
from exoplanet.stage3 import (
    STRUCTURAL_MAX_DEPTHS,
    build_shared_alphas,
    fold_agreement,
    select_configuration,
)


def test_development_frame_rejects_the_test_partition():
    frame = pd.DataFrame({"partition": ["development", "test"]})
    with pytest.raises(RuntimeError, match="test partition"):
        development_only(frame)


def test_primary_features_reject_the_significance_statistic():
    with pytest.raises(RuntimeError, match="koi_fwm_stat_sig"):
        assert_primary_features(["koi_period", "koi_fwm_stat_sig"])


def test_depth_grid_includes_the_stage2_neighborhood():
    for depth in (6, 7, 8, 9, 10, 12):
        assert depth in STRUCTURAL_MAX_DEPTHS


def test_selection_uses_validation_macro_f1_and_not_the_gap():
    weaker_gap = {
        "mean_validation_macro_f1": 0.80,
        "mean_macro_f1_gap": 0.01,
        "mean_n_leaves": 10,
    }
    higher_score = {
        "mean_validation_macro_f1": 0.81,
        "mean_macro_f1_gap": 0.20,
        "mean_n_leaves": 100,
    }
    assert select_configuration([weaker_gap, higher_score]) is higher_score


def test_fold_agreement_counts_direction_without_a_test():
    agreement = fold_agreement([0.02, -0.01, 0.0, 0.03, -0.04])
    assert agreement["n_higher"] == 2
    assert agreement["n_lower"] == 2
    assert agreement["n_equal"] == 1
    assert "p_value" not in agreement


def test_shared_alphas_come_from_training_paths_and_include_zero():
    paths = [
        np.asarray([0.0, 0.001, 0.01, 0.1]),
        np.asarray([0.0, 0.002, 0.02, 0.2]),
    ]
    alphas = build_shared_alphas(paths)
    assert alphas[0] == 0.0
    assert alphas == sorted(alphas)
    assert all(alpha >= 0 for alpha in alphas)
    assert len(alphas) > 1
