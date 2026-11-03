"""Search-design checks for the Stage 4 forest. No forest is trained here."""

from exoplanet.stage4 import (
    BASELINE,
    CLASS_WEIGHT_GRID,
    MAX_DEPTH_GRID,
    MAX_FEATURES_GRID,
    MIN_SAMPLES_LEAF_GRID,
    N_ESTIMATORS_GRID,
    forest_grid,
    forest_key,
    random_phase,
    select_forest,
)


def test_listed_grid_is_larger_than_the_random_phase():
    grid = forest_grid()
    assert len(grid) == (
        len(N_ESTIMATORS_GRID)
        * len(MAX_DEPTH_GRID)
        * len(MAX_FEATURES_GRID)
        * len(MIN_SAMPLES_LEAF_GRID)
        * len(CLASS_WEIGHT_GRID)
    )
    assert {row["max_depth"] for row in grid} == set(MAX_DEPTH_GRID)
    assert {row["max_features"] for row in grid} == set(MAX_FEATURES_GRID)
    assert {row["class_weight"] for row in grid} == set(CLASS_WEIGHT_GRID)
    sample = random_phase()
    assert len(sample) == 16
    assert len(sample) < len([row for row in grid if row["n_estimators"] == 100])
    assert random_phase() == sample
    assert all(row["n_estimators"] == 100 for row in sample)
    assert forest_key(BASELINE) not in {forest_key(row) for row in sample}


def test_forest_selection_uses_validation_macro_f1():
    smaller_gap = {
        "mean_validation_macro_f1": 0.70,
        "mean_macro_f1_gap": 0.01,
        "n_estimators": 100,
    }
    higher_score = {
        "mean_validation_macro_f1": 0.75,
        "mean_macro_f1_gap": 0.20,
        "n_estimators": 400,
    }
    assert select_forest([smaller_gap, higher_score]) is higher_score
