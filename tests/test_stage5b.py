"""Configuration rules for the Stage 5B sensitivity check. No model is trained."""

import pytest

from exoplanet.stage5b import LEAKAGE, build_configurations


def test_configurations_drop_centroids_and_mark_the_significance_column_diagnostic():
    primary = ["koi_period", "koi_dikco_msky", "koi_dikco_msky_err", "koi_model_snr"]
    centroids = ["koi_dikco_msky", "koi_dikco_msky_err"]
    configs = build_configurations(primary, centroids)
    assert configs["A"]["columns"] == primary
    assert configs["A"]["diagnostic"] is False
    assert configs["B"]["columns"] == ["koi_period", "koi_model_snr"]
    assert configs["B"]["diagnostic"] is False
    assert configs["C"]["columns"] == primary + ["koi_fwm_stat_sig"]
    assert configs["C"]["diagnostic"] is True
    assert "vetting-derived" in configs["C"]["role"]


def test_primary_set_rejects_leakage_and_the_diagnostic_column():
    with pytest.raises(RuntimeError):
        build_configurations(["koi_score", "koi_period"], [])
    with pytest.raises(RuntimeError):
        build_configurations(["koi_period", "koi_fwm_stat_sig"], [])
    assert "koi_fpflag_co" in LEAKAGE
