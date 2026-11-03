"""Run the Stage 1 inspection, audit, split, and preprocessing checks.

No classifier is trained.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import joblib
import pandas as pd

from exoplanet.audit import (
    build_feature_audit,
    dataset_overview,
    diagnostic_columns,
    experimental_stat_sig_columns,
    identify_unfitted_block,
    included_columns,
    profile_columns,
)
from exoplanet.config import (
    DATA_EXTERNAL,
    DATA_PROCESSED,
    DATASET_DOI,
    DATASET_SOURCE,
    DATASET_VERSION_DATE,
    DICTIONARY_RETRIEVED,
    DOCS,
    FIGURES,
    GROUP_COLUMN,
    HOLDOUT_FOLD_INDEX,
    HOLDOUT_N_SPLITS,
    ID_COLUMN,
    KOI_2012_DEFINITIONS_URL,
    KOI_DICTIONARY_HTML,
    KOI_DICTIONARY_UPDATED,
    KOI_DICTIONARY_URL,
    RANDOM_SEED,
    SAVED_SPLIT_SHA256,
    RAW_CSV,
    RESULTS,
    TCE_DICTIONARY_HTML,
    TCE_DICTIONARY_URL,
)
from exoplanet.dictionary import odd_even_documentation, parse_column_page
from exoplanet.eda import save_figures
from exoplanet.io import ensure_raw_copy, load_koi
from exoplanet.preprocess import (
    assert_imputer_uses_training_rows,
    diagnostic_learning_frame,
    fit_imputer,
    primary_learning_frame,
    training_fold_index,
)
from exoplanet.reporting import write_findings
from exoplanet.splits import assert_split_integrity, assign_splits, split_balance


def main() -> None:
    for directory in (DATA_EXTERNAL, DATA_PROCESSED, RESULTS, FIGURES, DOCS):
        directory.mkdir(parents=True, exist_ok=True)

    file_record = ensure_raw_copy()
    frame = load_koi(RAW_CSV).reset_index(drop=True)
    documents = parse_column_page(KOI_DICTIONARY_HTML)
    tce_documents = parse_column_page(TCE_DICTIONARY_HTML)
    _assert_dictionary(documents, tce_documents)
    odd_even = odd_even_documentation(tce_documents)

    overview = dataset_overview(frame)
    profile = profile_columns(frame)
    block = identify_unfitted_block(frame)
    audit = build_feature_audit(frame, documents, odd_even, block)
    features = included_columns(audit)
    experimental = experimental_stat_sig_columns(audit)
    primary_row = ~block["row_mask"]
    splits = assign_splits(frame, primary_row)
    repeated = assign_splits(frame, primary_row)
    if not splits.equals(repeated):
        raise RuntimeError("The grouped split changed when repeated with seed 245.")
    splits = _retain_saved_split(splits)
    assert_split_integrity(splits)
    balance = split_balance(frame, splits)

    primary = primary_learning_frame(frame, splits, features)
    diagnostic = diagnostic_learning_frame(frame, primary)
    _assert_matrices(primary, diagnostic, features, block)

    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    train_index = training_fold_index(splits, fold_id=0)
    pipeline = fit_imputer(numeric, splits["partition"], train_index)
    assert_imputer_uses_training_rows(numeric, pipeline, train_index)
    test_index = splits.index[splits["partition"].eq("test")]
    if not set(train_index).isdisjoint(test_index):
        raise RuntimeError("The imputer training index includes test rows.")

    figures = save_figures(frame, block["row_mask"], features, FIGURES)
    _write_outputs(
        file_record,
        frame,
        documents,
        overview,
        profile,
        block,
        audit,
        features,
        experimental,
        splits,
        balance,
        primary,
        diagnostic,
        train_index,
        figures,
    )
    print(
        f"Stage 1 complete: {overview['n_rows']} rows, "
        f"{len(features)} proposed features, "
        f"{block['n_unfitted_rows']} rows without transit depth, "
        f"test groups {balance['test']['n_groups']}, overlap {balance['group_overlap']}."
    )


def _assert_dictionary(documents: dict, tce_documents: dict) -> None:
    if "Robovetter" not in documents["koi_score"]["description"]:
        raise RuntimeError("koi_score was not parsed from the KOI column page.")
    if "koi_period_err1" not in documents:
        raise RuntimeError("Period uncertainties were not parsed.")
    if "Centroid Offset Flag" not in documents["koi_fpflag_co"]["label"]:
        raise RuntimeError("koi_fpflag_co was not parsed.")
    if "tce_bin_oedp_sig" not in tce_documents:
        raise RuntimeError("The TCE odd-even column was not parsed.")


def _assert_matrices(primary, diagnostic, features, block) -> None:
    if primary["koi_depth"].isna().any():
        raise RuntimeError("Primary matrix contains rows without transit depth.")
    if len(primary) != int((~block["row_mask"]).sum()):
        raise RuntimeError("Primary matrix row count does not match the depth filter.")
    leaked = {"koi_score", "koi_pdisposition", "koi_fpflag_co", "kepler_name", "kepid", "koi_fwm_stat_sig"}
    if leaked & set(features):
        raise RuntimeError(f"Leakage or identifier columns entered the feature list: {leaked & set(features)}")
    if "koi_score" not in diagnostic.columns or "has_kepler_name" not in diagnostic.columns:
        raise RuntimeError("Diagnostic matrix is missing its leakage columns.")
    if "koi_score" in primary.columns:
        raise RuntimeError("Disposition score entered the scientific matrix.")


_SPLIT_COLUMNS = [
    "kepoi_name",
    "kepid",
    "koi_disposition",
    "koi_fittype",
    "partition",
    "dev_fold",
    "primary_supervised_row",
]


def _normalize_split(table: pd.DataFrame) -> pd.DataFrame:
    out = table[_SPLIT_COLUMNS].copy()
    out["kepoi_name"] = out["kepoi_name"].astype(str)
    out["kepid"] = pd.to_numeric(out["kepid"]).astype("int64")
    out["dev_fold"] = pd.to_numeric(out["dev_fold"]).astype("int64")
    out["primary_supervised_row"] = out["primary_supervised_row"].map(
        lambda value: value if isinstance(value, (bool, np.bool_)) else str(value).lower() in {"true", "1"}
    )
    for column in ("koi_disposition", "koi_fittype", "partition"):
        out[column] = out[column].astype(str)
    return out.reset_index(drop=True)


def _retain_saved_split(fresh: pd.DataFrame) -> pd.DataFrame:
    """Use the saved seed-245 assignments. Do not write a new split."""
    path = DATA_PROCESSED / "proposed_splits.csv"
    if not path.exists():
        raise RuntimeError(
            "data/processed/proposed_splits.csv is missing. "
            "Refusing to create a new grouped split."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SAVED_SPLIT_SHA256:
        raise RuntimeError(
            "data/processed/proposed_splits.csv does not match the saved "
            f"seed-245 split (expected SHA-256 {SAVED_SPLIT_SHA256}). "
            "The file was not replaced."
        )
    saved = pd.read_csv(path)
    if not _normalize_split(saved).equals(_normalize_split(fresh)):
        raise RuntimeError(
            "Recomputing the seed-245 grouped split does not match "
            "data/processed/proposed_splits.csv. The saved file was not replaced."
        )
    aligned = saved.copy()
    aligned.index = fresh.index
    aligned["primary_supervised_row"] = aligned["primary_supervised_row"].astype(bool)
    aligned["dev_fold"] = aligned["dev_fold"].astype(int)
    return aligned


def _write_outputs(
    file_record,
    frame,
    documents,
    overview,
    profile,
    block,
    audit,
    features,
    experimental,
    splits,
    balance,
    primary,
    diagnostic,
    train_index,
    figures,
) -> None:
    record = {
        **file_record,
        "source": DATASET_SOURCE,
        "doi": DATASET_DOI,
        "version_date": DATASET_VERSION_DATE,
        "n_rows": overview["n_rows"],
        "n_columns": overview["n_columns"],
        "dictionary_url": KOI_DICTIONARY_URL,
        "dictionary_retrieved": DICTIONARY_RETRIEVED,
        "dictionary_page_updated": KOI_DICTIONARY_UPDATED,
        "tce_dictionary_url": TCE_DICTIONARY_URL,
        "koi_2012_definitions_url": KOI_2012_DEFINITIONS_URL,
        "random_seed": RANDOM_SEED,
        "holdout": {
            "method": "StratifiedGroupKFold",
            "n_splits": HOLDOUT_N_SPLITS,
            "fold_index": HOLDOUT_FOLD_INDEX,
            "shuffle": True,
            "group": "kepid",
        },
        "team_finalized": False,
    }
    _json(RESULTS / "dataset_record.json", record)
    _json(DATA_PROCESSED / "proposal_status.json", {
        "status": "proposed",
        "team_finalized": False,
        "prepared_by": "Danish Waseem",
        "pending_agreement_from": [
            "Sanjana Reddy Katham",
            "Nathan Pak",
            "Venkata Sai Bhavana Yarrapragada",
        ],
        "random_seed": RANDOM_SEED,
        "note": (
            "Canonical split and feature list proposed for the team. "
            "Do not treat them as final until the other members agree. "
            "koi_fwm_stat_sig is excluded from the primary set and saved only "
            "in the experimental configuration. The 363 KOIs without transit "
            "depth are excluded from the primary experiment and retain their "
            "original split assignments for a later sensitivity experiment."
        ),
        "primary_feature_count": len(features),
        "fwm_stat_sig_in_primary": "koi_fwm_stat_sig" in features,
        "incomplete_kois_retained_for_sensitivity": int((~splits["primary_supervised_row"]).sum()),
        "permanently_discarded_incomplete_kois": False,
    })
    profile.to_csv(RESULTS / "column_profile.csv", index=False)
    audit.drop(columns=["description"]).to_csv(RESULTS / "column_audit.csv", index=False)
    audit.loc[audit["decision"] == "include", [
        "column", "feature_family", "provisional", "impute_residual",
        "missing_fraction_primary_rows", "dictionary_label", "reason",
    ]].to_csv(DATA_PROCESSED / "proposed_leakage_aware_features.csv", index=False)
    pd.DataFrame({"column": diagnostic_columns(audit)}).to_csv(
        DATA_PROCESSED / "proposed_diagnostic_features.csv", index=False
    )
    if experimental != features + ["koi_fwm_stat_sig"]:
        raise RuntimeError("The experimental feature list is not the primary list plus koi_fwm_stat_sig.")
    feature_table = audit.set_index("column")
    experimental_rows = feature_table.loc[experimental, [
        "feature_family", "provisional", "dictionary_label", "reason",
    ]].reset_index()
    experimental_rows.insert(1, "role", ["primary"] * len(features) + ["experimental_only"])
    experimental_rows.to_csv(
        DATA_PROCESSED / "experimental_features_with_fwm_stat_sig.csv",
        index=False,
    )
    retained = splits.loc[~splits["primary_supervised_row"].astype(bool)].copy()
    retained["excluded_from_primary_experiment"] = True
    retained["permanently_discarded"] = False
    retained["purpose"] = "later_sensitivity_experiment"
    retained.to_csv(DATA_PROCESSED / "incomplete_koi_split_assignments.csv", index=False)
    # The grouped split file is verified against seed 245 and is not rewritten.
    primary.to_csv(DATA_PROCESSED / "proposed_primary_matrix_unimputed.csv", index=False)
    diagnostic.to_csv(DATA_PROCESSED / "proposed_diagnostic_matrix_unimputed.csv", index=False)
    pd.DataFrame(
        [{"column": name, **{key: value for key, value in doc.items() if key != "uncertainty_columns"}}
         | {"uncertainty_columns": ",".join(doc.get("uncertainty_columns", []))}
         for name, doc in sorted(documents.items())]
    ).to_csv(DATA_EXTERNAL / "koi_columns_parsed.csv", index=False)

    block_public = {key: value for key, value in block.items() if key != "row_mask"}
    _json(RESULTS / "unfitted_block.json", block_public)
    _json(RESULTS / "split_balance.json", balance)
    _json(RESULTS / "overview.json", overview)
    joblib.dump(
        {
            "random_seed": RANDOM_SEED,
            "team_finalized": False,
            "feature_columns": features,
            "test_kepid": sorted(set(splits.loc[splits["partition"].eq("test"), GROUP_COLUMN])),
            "development_kepid": sorted(
                set(splits.loc[splits["partition"].eq("development"), GROUP_COLUMN])
            ),
            "development_fold0_train_kepoi_name": splits.loc[train_index, ID_COLUMN].astype(str).tolist(),
            "figures": figures,
        },
        RESULTS / "proposed_split.joblib",
    )
    write_findings(
        DOCS / "stage1_findings.md",
        record,
        overview,
        block_public,
        audit,
        balance,
        audit.loc[audit["impute_residual"].astype(bool), "column"].tolist(),
    )


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
