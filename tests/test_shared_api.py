"""Shared development API. Does not train teammate classifiers."""

from sklearn.pipeline import Pipeline

from exoplanet.config import DATA_PROCESSED, RAW_CSV, SAVED_SPLIT_SHA256
from exoplanet.development import load_development_bundle, load_feature_list, load_splits
from exoplanet.io import file_sha256
from exoplanet.preprocess import make_imputer, make_preprocess_pipeline


def test_dataset_and_split_checksums():
    assert (
        file_sha256(RAW_CSV)
        == "a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab"
    )
    splits = load_splits()
    assert len(splits) == 9564
    assert file_sha256(DATA_PROCESSED / "proposed_splits.csv") == SAVED_SPLIT_SHA256


def test_development_bundle_excludes_test_and_keeps_primary_shape():
    bundle = load_development_bundle()
    assert bundle.n_rows == 7381
    assert len(bundle.feature_columns) == 76
    assert set(bundle.frame["partition"].unique()) == {"development"}
    assert bundle.split_sha256 == SAVED_SPLIT_SHA256
    assert "koi_fwm_stat_sig" not in bundle.feature_columns


def test_feature_list_length():
    assert len(load_feature_list()) == 76


def test_preprocess_scaling_flag():
    plain = make_preprocess_pipeline(scale=False)
    scaled = make_preprocess_pipeline(scale=True)
    assert isinstance(plain, Pipeline)
    assert list(plain.named_steps) == ["impute"]
    assert list(scaled.named_steps) == ["impute", "scale"]
    assert list(make_imputer().named_steps) == ["impute"]
