"""Shared development-partition loader for parallel teammate experiments.

Loads the proposed 76-feature primary matrix and the locked grouped folds.
The test partition is never returned. Existing Stage 1–5B artifacts are not
rewritten by this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from exoplanet.config import (
    CLASS_ORDER,
    DATA_PROCESSED,
    GROUP_COLUMN,
    ID_COLUMN,
    RANDOM_SEED,
    SAVED_SPLIT_SHA256,
    TARGET_COLUMN,
)
from exoplanet.io import file_sha256
from exoplanet.stage2 import _load_inputs, assert_primary_features, development_only


@dataclass(frozen=True)
class DevelopmentBundle:
    """Development rows ready for grouped cross-validation."""

    frame: pd.DataFrame
    feature_columns: list[str]
    split_sha256: str
    n_rows: int
    n_groups: int
    class_order: tuple[str, ...]
    random_seed: int
    proposal_status: str


def load_feature_list(path=None) -> list[str]:
    """Return the proposed 76-column primary feature list."""
    table = pd.read_csv(path or (DATA_PROCESSED / "proposed_leakage_aware_features.csv"))
    columns = table["column"].astype(str).tolist()
    assert_primary_features(columns)
    if len(columns) != 76:
        raise RuntimeError(f"Expected 76 primary features, found {len(columns)}.")
    return columns


def load_splits() -> pd.DataFrame:
    """Load the locked split file and refuse a changed checksum."""
    path = DATA_PROCESSED / "proposed_splits.csv"
    digest = file_sha256(path)
    if digest != SAVED_SPLIT_SHA256:
        raise RuntimeError(
            "proposed_splits.csv changed. Expected "
            f"{SAVED_SPLIT_SHA256}, found {digest}."
        )
    return pd.read_csv(path)


def load_development_bundle() -> DevelopmentBundle:
    """Load development primary rows, features, and the split checksum.

    Returns only partition == development rows with a measured transit depth.
    The test partition is not included. Do not score the test set from this
    bundle during model development.
    """
    frame, columns, digest = _load_inputs()
    frame = development_only(frame)
    status = json.loads((DATA_PROCESSED / "proposal_status.json").read_text(encoding="utf-8"))
    return DevelopmentBundle(
        frame=frame,
        feature_columns=columns,
        split_sha256=digest,
        n_rows=len(frame),
        n_groups=int(frame[GROUP_COLUMN].nunique()),
        class_order=tuple(CLASS_ORDER),
        random_seed=RANDOM_SEED,
        proposal_status=str(status.get("status", "unknown")),
    )


def development_fold_masks(frame: pd.DataFrame, fold_id: int) -> tuple[pd.Series, pd.Series]:
    """Boolean masks for the saved development training and validation fold."""
    frame = development_only(frame)
    if fold_id not in range(5):
        raise ValueError(f"fold_id must be 0..4, got {fold_id}.")
    folds = frame["dev_fold"]
    train = folds.ne(fold_id)
    valid = folds.eq(fold_id)
    train_groups = set(frame.loc[train, GROUP_COLUMN])
    valid_groups = set(frame.loc[valid, GROUP_COLUMN])
    if train_groups & valid_groups:
        raise RuntimeError(f"Fold {fold_id} shares a host star with its training rows.")
    return train, valid


def feature_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return the modeling feature columns only."""
    assert_primary_features(columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Frame is missing features: {missing}")
    return frame[columns]


def labels(frame: pd.DataFrame) -> pd.Series:
    """Return the disposition labels."""
    return frame[TARGET_COLUMN]


def assert_no_test_rows(frame: pd.DataFrame) -> None:
    """Refuse any frame that contains the test partition."""
    development_only(frame)


def split_identity_record() -> dict:
    """Small record teammates can save with their experiment outputs."""
    splits = load_splits()
    features = load_feature_list()
    return {
        "split_sha256": file_sha256(DATA_PROCESSED / "proposed_splits.csv"),
        "expected_split_sha256": SAVED_SPLIT_SHA256,
        "n_split_rows": int(len(splits)),
        "n_primary_features": len(features),
        "feature_list": "data/processed/proposed_leakage_aware_features.csv",
        "split_file": "data/processed/proposed_splits.csv",
        "id_column": ID_COLUMN,
        "group_column": GROUP_COLUMN,
        "target_column": TARGET_COLUMN,
        "class_order": list(CLASS_ORDER),
        "random_seed": RANDOM_SEED,
        "test_scored": False,
    }
