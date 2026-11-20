"""Setups and feature sets for the Person 1 experiments.

Everything starts from the shared development bundle, which contains only
partition == development rows with a measured transit depth. Nothing here
reads or returns test-partition rows for modeling.
"""

from __future__ import annotations

import pandas as pd

from exoplanet.config import CLASS_ORDER, DATA_PROCESSED, ID_COLUMN, RAW_CSV, TARGET_COLUMN
from exoplanet.development import DevelopmentBundle, assert_no_test_rows
from exoplanet.io import load_koi
from exoplanet.preprocess import diagnostic_learning_frame

SETUPS = ("3class", "binary")
BINARY_CLASSES = ("CONFIRMED", "FALSE POSITIVE")
POSITIVE = "CONFIRMED"

# Disposition and vetting outputs. Used only in the E3 leakage diagnostic,
# never as real features. pdisp_is_candidate encodes koi_pdisposition and
# has_kepler_name encodes kepler_name (see exoplanet.preprocess).
LEAKAGE_COLUMNS = [
    "pdisp_is_candidate",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "has_kepler_name",
]


def class_order(setup: str) -> tuple[str, ...]:
    return tuple(CLASS_ORDER) if setup == "3class" else BINARY_CLASSES


def setup_frame(frame: pd.DataFrame, setup: str) -> tuple[pd.DataFrame, pd.Series]:
    """Rows and labels for a setup.

    3class keeps every development row. binary drops CANDIDATE rows and keeps
    the same saved dev_fold values, so the binary folds are the 3-class folds
    with CANDIDATE removed.
    """
    assert_no_test_rows(frame)
    if setup == "3class":
        rows = frame
    elif setup == "binary":
        rows = frame.loc[frame[TARGET_COLUMN].ne("CANDIDATE")]
    else:
        raise ValueError(setup)
    return rows, rows[TARGET_COLUMN].astype(str)


def feature_sets(bundle: DevelopmentBundle) -> dict[str, list[str]]:
    """F76 (primary), F56 (no provisional centroid columns), F76 plus leakage."""
    table = pd.read_csv(DATA_PROCESSED / "proposed_leakage_aware_features.csv")
    provisional = set(table.loc[table["provisional"].astype(bool), "column"])
    f76 = list(bundle.feature_columns)
    f56 = [c for c in f76 if c not in provisional]
    if len(f56) != 56:
        raise RuntimeError(f"Expected 56 non-centroid features, found {len(f56)}.")
    return {"F76": f76, "F56": f56, "F76_plus_leakage": f76 + LEAKAGE_COLUMNS}


def add_leakage_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the diagnostic disposition columns to the development rows.

    Uses the shared diagnostic_learning_frame on the raw KOI table. Rows are
    matched to the raw table through kepoi_name, which is unique there.
    """
    assert_no_test_rows(frame)
    raw = load_koi(RAW_CSV)
    if raw[ID_COLUMN].duplicated().any():
        raise RuntimeError("kepoi_name is not unique in the raw KOI table.")
    position = pd.Series(raw.index, index=raw[ID_COLUMN].astype(str))
    raw_index = position.loc[frame[ID_COLUMN].astype(str)].to_numpy()
    keyed = frame.set_axis(raw_index, axis=0)
    diag = diagnostic_learning_frame(raw, keyed)
    if not diag[ID_COLUMN].astype(str).equals(keyed[ID_COLUMN].astype(str)):
        raise RuntimeError("Leakage columns did not align with the development rows.")
    out = frame.copy()
    out["pdisp_is_candidate"] = diag["koi_pdisposition"].eq("CANDIDATE").astype(int).to_numpy()
    for column in LEAKAGE_COLUMNS[1:]:
        out[column] = diag[column].to_numpy()
    return out
