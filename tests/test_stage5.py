"""Rules for the Stage 5 importance summary. No model is trained."""

from exoplanet.stage5 import centroid_dominates, ranking_is_stable


def test_centroid_rule_flags_a_leading_offset_or_a_large_share():
    centroid = {"koi_dicco_msky"}
    assert centroid_dominates(["koi_dicco_msky", "koi_period"], centroid, 0.1)
    assert not centroid_dominates(["koi_period", "koi_depth"], centroid, 0.1)
    assert centroid_dominates(["koi_period"] * 10, centroid, 0.5)


def test_stability_rule_requires_both_correlation_and_rank_consistency():
    assert ranking_is_stable(0.9, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert not ranking_is_stable(0.79, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert not ranking_is_stable(0.9, [20, 20, 20, 1, 1, 1, 1, 1, 1, 1])
