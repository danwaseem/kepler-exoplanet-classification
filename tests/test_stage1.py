"""Stage 1 data-preparation tests. No classifier is trained."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exoplanet.audit import (
    build_feature_audit,
    experimental_stat_sig_columns,
    identify_unfitted_block,
    included_columns,
)
from exoplanet.config import (
    KOI_DICTIONARY_HTML,
    LEAKAGE_COLUMNS,
    RANDOM_SEED,
    RAW_CSV,
    SOURCE_CSV,
    TCE_DICTIONARY_HTML,
)
from exoplanet.dictionary import odd_even_documentation, parse_column_page
from exoplanet.io import file_sha256, load_koi
from exoplanet.preprocess import fit_imputer, primary_learning_frame, training_fold_index
from exoplanet.splits import assert_split_integrity, assign_splits

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared():
    frame = load_koi(RAW_CSV).reset_index(drop=True)
    documents = parse_column_page(KOI_DICTIONARY_HTML)
    odd_even = odd_even_documentation(parse_column_page(TCE_DICTIONARY_HTML))
    block = identify_unfitted_block(frame)
    audit = build_feature_audit(frame, documents, odd_even, block)
    splits = assign_splits(frame, ~block["row_mask"])
    return {
        "frame": frame,
        "documents": documents,
        "block": block,
        "audit": audit,
        "splits": splits,
    }


def test_dictionary_identifies_disposition_and_uncertainties():
    documents = parse_column_page(KOI_DICTIONARY_HTML)
    assert "Robovetter" in documents["koi_score"]["description"]
    assert "koi_period_err1" in documents["koi_period"]["uncertainty_columns"]
    assert documents["koi_fpflag_co"]["label"] == "Centroid Offset Flag"
    assert "difference image" in documents["koi_dicco_msky"]["description"].lower()
    tce = parse_column_page(TCE_DICTIONARY_HTML)
    assert "false positive" in tce["tce_bin_oedp_sig"]["description"].lower()


def test_raw_csv_is_unchanged():
    """Canonical copy is data/raw/. A root-level duplicate is optional."""
    assert RAW_CSV.exists()
    assert file_sha256(RAW_CSV) == (
        "a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab"
    )
    if SOURCE_CSV.exists():
        assert file_sha256(SOURCE_CSV) == file_sha256(RAW_CSV)


def test_split_keeps_host_stars_together_and_repeats(prepared):
    splits = prepared["splits"]
    assert_split_integrity(splits)
    again = assign_splits(prepared["frame"], ~prepared["block"]["row_mask"])
    assert splits.equals(again)
    assert RANDOM_SEED == 245
    test_groups = set(splits.loc[splits["partition"].eq("test"), "kepid"])
    dev_groups = set(splits.loc[splits["partition"].eq("development"), "kepid"])
    assert test_groups.isdisjoint(dev_groups)
    fraction = len(test_groups) / splits["kepid"].nunique()
    assert 0.15 <= fraction <= 0.25


def test_feature_policy_is_not_a_blanket_uncertainty_drop(prepared):
    audit = prepared["audit"].set_index("column")
    included = set(included_columns(prepared["audit"]))
    for column in LEAKAGE_COLUMNS:
        assert audit.loc[column, "decision"] == "exclude"
    assert "kepid" not in included
    assert "koi_disposition" not in included
    errors = prepared["audit"][prepared["audit"]["kind"] == "uncertainty"]
    assert (errors["decision"] == "include").any()
    assert (errors["decision"] == "exclude").any()
    assert audit.loc["koi_period_err1", "decision"] == "include"
    assert audit.loc["koi_teq_err1", "decision"] == "exclude"
    assert audit.loc["koi_dicco_msky", "decision"] == "include"
    assert bool(audit.loc["koi_dicco_msky", "provisional"])
    assert audit.loc["koi_fwm_stat_sig", "decision"] == "exclude"
    assert "koi_fwm_stat_sig" not in included
    experimental = experimental_stat_sig_columns(prepared["audit"])
    assert experimental[-1] == "koi_fwm_stat_sig"
    assert experimental[:-1] == included_columns(prepared["audit"])
    assert audit.loc["koi_fwm_sra", "decision"] == "exclude"
    assert audit.loc["koi_fpflag_co", "decision"] == "exclude"
    assert audit.loc["koi_time0", "decision"] == "exclude"
    assert audit.loc["koi_time0bk", "decision"] == "include"


def test_unfitted_rows_are_omitted_and_not_required_for_depth(prepared):
    frame = prepared["frame"]
    block = prepared["block"]
    unfitted = block["row_mask"]
    assert block["n_unfitted_rows"] == int(unfitted.sum())
    assert frame.loc[unfitted, "koi_depth"].isna().all()
    assert frame.loc[~unfitted, "koi_depth"].notna().all()
    assert frame.loc[unfitted, "koi_period"].notna().all()
    primary = primary_learning_frame(
        frame,
        prepared["splits"],
        included_columns(prepared["audit"]),
    )
    assert primary["koi_depth"].notna().all()
    assert len(primary) == int((~unfitted).sum())


def test_imputer_uses_only_development_training_rows(prepared):
    features = pd.DataFrame({"a": [1.0, np.nan, 5.0, 100.0]})
    partition = pd.Series(["development", "development", "development", "test"])
    pipeline = fit_imputer(features, partition, features.index[:2])
    assert pipeline.named_steps["impute"].statistics_[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fit_imputer(features, partition, features.index[[0, 3]])

    frame = prepared["frame"]
    splits = prepared["splits"]
    columns = included_columns(prepared["audit"])
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    train_index = training_fold_index(splits, fold_id=0)
    assert set(splits.loc[train_index, "partition"]) == {"development"}
    assert splits.loc[train_index, "primary_supervised_row"].all()
    fitted = fit_imputer(numeric, splits["partition"], train_index)
    expected = np.nanmedian(numeric.loc[train_index].to_numpy(dtype=float), axis=0)
    assert np.allclose(fitted.named_steps["impute"].statistics_, expected, equal_nan=True)


def test_saved_split_retains_incomplete_kois():
    saved = pd.read_csv(ROOT / "data" / "processed" / "proposed_splits.csv")
    retained = pd.read_csv(ROOT / "data" / "processed" / "incomplete_koi_split_assignments.csv")
    incomplete = saved.loc[~saved["primary_supervised_row"].astype(bool)]
    assert len(incomplete) == 363
    assert len(retained) == 363
    assert retained["permanently_discarded"].eq(False).all()
    assert set(retained["partition"]) <= {"development", "test"}
    features = pd.read_csv(ROOT / "data" / "processed" / "proposed_leakage_aware_features.csv")
    assert "koi_fwm_stat_sig" not in set(features["column"])
    experimental = pd.read_csv(
        ROOT / "data" / "processed" / "experimental_features_with_fwm_stat_sig.csv"
    )
    assert experimental.loc[experimental["role"].eq("experimental_only"), "column"].tolist() == [
        "koi_fwm_stat_sig"
    ]


def test_source_does_not_train_other_classifiers():
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in ("LogisticRegression", "KNeighbors", "SVC"):
            assert name not in text, path.name
        if path.name != "stage4.py":
            assert "RandomForest" not in text, path.name
        if path.name not in {"stage2.py", "stage3.py"}:
            assert "DecisionTree" not in text, path.name
