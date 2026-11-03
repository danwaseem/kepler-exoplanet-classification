#!/usr/bin/env python3
"""Check the shared table, the 76-feature list, and the saved grouped split.

Uses the canonical files in the repository root when this package lives inside
the main project (``data/raw/`` and ``data/processed/``). Duplicate CSV copies
inside ``team_shared_setup/`` are optional and are not required after a clone.

This script only reads files. It does not write a new split or score the test set.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
RAW_FILENAME = "koi_cumulative_2026-10-06.csv"
CANONICAL_RAW = PROJECT_ROOT / "data" / "raw" / RAW_FILENAME

# Prefer the main repository package and paths after a fresh clone.
if CANONICAL_RAW.is_file():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    WORK_ROOT = PROJECT_ROOT
else:
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))
    WORK_ROOT = PACKAGE_ROOT

import pandas as pd

from exoplanet.config import (
    DATA_PROCESSED,
    GROUP_COLUMN,
    LEAKAGE_COLUMNS,
    RAW_CSV,
    RESULTS,
    SAVED_SPLIT_SHA256,
    SOURCE_CSV,
    TARGET_COLUMN,
)
from exoplanet.io import file_sha256, load_koi
from exoplanet.splits import assert_split_integrity

EXPECTED_CSV_SHA256 = "a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab"
EXPECTED_FEATURE_COUNT = 76
EXPECTED_INCOMPLETE_ROWS = 363
FORBIDDEN_FEATURES = set(LEAKAGE_COLUMNS) | {
    TARGET_COLUMN,
    GROUP_COLUMN,
    "koi_fwm_stat_sig",
    "kepoi_name",
}


def _audit_path() -> Path:
    candidates = [
        DATA_PROCESSED / "column_audit.csv",
        RESULTS / "column_audit.csv",
        WORK_ROOT / "results" / "shared" / "column_audit.csv",
        PACKAGE_ROOT / "data" / "processed" / "column_audit.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def main() -> None:
    failures: list[str] = []

    def check(passed: bool, message: str) -> None:
        print(("PASS  " if passed else "FAIL  ") + message)
        if not passed:
            failures.append(message)

    check(RAW_CSV.is_file(), f"canonical NASA CSV is present at {RAW_CSV}")
    raw_hash = file_sha256(RAW_CSV) if RAW_CSV.is_file() else ""
    check(raw_hash == EXPECTED_CSV_SHA256, f"NASA CSV checksum is {EXPECTED_CSV_SHA256}")
    if SOURCE_CSV.is_file():
        source_hash = file_sha256(SOURCE_CSV)
        check(
            source_hash == EXPECTED_CSV_SHA256,
            "optional workspace CSV copy matches the expected checksum",
        )
        check(source_hash == raw_hash, "optional workspace CSV matches data/raw")
    else:
        print("INFO  optional root-level NASA CSV copy is absent (expected after clone)")

    split_path = DATA_PROCESSED / "proposed_splits.csv"
    feature_path = DATA_PROCESSED / "proposed_leakage_aware_features.csv"
    audit_path = _audit_path()
    check(split_path.is_file(), "proposed_splits.csv is present")
    check(feature_path.is_file(), "proposed_leakage_aware_features.csv is present")
    split_hash = file_sha256(split_path) if split_path.is_file() else ""
    check(split_hash == SAVED_SPLIT_SHA256, f"split checksum is {SAVED_SPLIT_SHA256}")

    frame = load_koi(RAW_CSV).reset_index(drop=True)
    features = pd.read_csv(feature_path)
    columns = features["column"].astype(str).tolist()
    check(len(columns) == EXPECTED_FEATURE_COUNT, f"feature list has {EXPECTED_FEATURE_COUNT} columns")
    check(len(columns) == len(set(columns)), "feature names are unique")
    missing = [column for column in columns if column not in frame.columns]
    check(not missing, "every proposed feature is present in the NASA table")
    forbidden = sorted(set(columns) & FORBIDDEN_FEATURES)
    check(not forbidden, "leakage columns, identifiers, and koi_fwm_stat_sig are absent")
    provisional = set(features.loc[features["provisional"].astype(bool), "column"].astype(str))
    check("koi_dicco_msky" in provisional, "koi_dicco_msky remains a provisional centroid feature")

    splits = pd.read_csv(split_path)
    check(len(splits) == len(frame), "the split file has one row per KOI")
    check(
        splits["kepoi_name"].astype(str).tolist() == frame["kepoi_name"].astype(str).tolist(),
        "split rows stay in the same order as the NASA table",
    )
    try:
        assert_split_integrity(splits)
        integrity_ok = True
    except AssertionError as exc:
        integrity_ok = False
        print(f"      {exc}")
    check(integrity_ok, "saved split passes the integrity checks")
    test_groups = set(splits.loc[splits["partition"].eq("test"), GROUP_COLUMN])
    dev_groups = set(splits.loc[splits["partition"].eq("development"), GROUP_COLUMN])
    check(test_groups.isdisjoint(dev_groups), "no host star is in both development and test")

    development = splits.loc[splits["partition"].eq("development") & splits["primary_supervised_row"]]
    fold_ids = set(development["dev_fold"].astype(int))
    check(fold_ids == set(range(5)), "development primary rows use saved CV folds 0-4")
    groups_per_fold = development.groupby(GROUP_COLUMN)["dev_fold"].nunique()
    check(bool((groups_per_fold == 1).all()), "each development host star has one CV fold")

    incomplete = ~splits["primary_supervised_row"].astype(bool)
    check(int(incomplete.sum()) == EXPECTED_INCOMPLETE_ROWS, "363 incomplete KOIs remain in the split file")
    check(
        set(splits.loc[incomplete, "partition"]) <= {"development", "test"},
        "incomplete KOIs still have development or test assignments",
    )
    depth_missing = frame["koi_depth"].isna()
    check(
        bool(depth_missing.equals(incomplete)),
        "primary rows are exactly the rows with a measured transit depth",
    )
    check(
        audit_path.is_file()
        and "koi_fpflag_co" in pd.read_csv(audit_path)["column"].astype(str).tolist(),
        "column leakage audit is present",
    )

    if failures:
        print(f"Verification failed: {len(failures)} check(s).")
        sys.exit(1)
    print(f"Verification passed (using {WORK_ROOT}).")


if __name__ == "__main__":
    main()
