"""Grouped development/test split and development-set cross-validation folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from exoplanet.config import (
    CLASS_ORDER,
    DEV_CV_N_SPLITS,
    GROUP_COLUMN,
    HOLDOUT_FOLD_INDEX,
    HOLDOUT_N_SPLITS,
    ID_COLUMN,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from exoplanet.audit import disposition_counts, disposition_rates


def assign_splits(frame: pd.DataFrame, primary_row: pd.Series) -> pd.DataFrame:
    """Hold out one stratified group fold, then fold the development stars.

    StratifiedGroupKFold keeps every KOI of a kepid in the same partition and
    tries to preserve class proportions. The test fold is fold index 0 of a
    seeded 5-fold split, which is the 20% design. Tuning folds are a new
    5-fold split fit only on the development stars. The test set is not one
    of those tuning folds.
    """
    if not primary_row.index.equals(frame.index):
        raise ValueError("primary_row must be aligned to the KOI table.")
    y = frame[TARGET_COLUMN].to_numpy()
    groups = frame[GROUP_COLUMN].to_numpy()
    holdout = StratifiedGroupKFold(
        n_splits=HOLDOUT_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    folds = list(holdout.split(np.zeros(len(frame)), y, groups))
    _development, test_pos = folds[HOLDOUT_FOLD_INDEX]
    partition = np.full(len(frame), "development", dtype=object)
    partition[test_pos] = "test"

    dev_fold = np.full(len(frame), -1, dtype=int)
    dev_pos = np.flatnonzero(partition == "development")
    inner = StratifiedGroupKFold(
        n_splits=DEV_CV_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    dev_frame = frame.iloc[dev_pos]
    for fold_id, (_train_rel, val_rel) in enumerate(
        inner.split(
            np.zeros(len(dev_frame)),
            dev_frame[TARGET_COLUMN].to_numpy(),
            dev_frame[GROUP_COLUMN].to_numpy(),
        )
    ):
        dev_fold[dev_pos[val_rel]] = fold_id

    assigned = pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN].astype(str).to_numpy(),
            GROUP_COLUMN: frame[GROUP_COLUMN].to_numpy(),
            TARGET_COLUMN: frame[TARGET_COLUMN].astype(str).to_numpy(),
            "koi_fittype": frame["koi_fittype"].astype(str).to_numpy(),
            "partition": partition,
            "dev_fold": dev_fold,
            "primary_supervised_row": primary_row.to_numpy(),
        },
        index=frame.index,
    )
    return assigned


def split_balance(frame: pd.DataFrame, splits: pd.DataFrame) -> dict:
    """Class mix of the full table, each partition, and the primary-row subset."""
    full_counts = disposition_counts(frame[TARGET_COLUMN])
    full_rates = disposition_rates(full_counts)
    report = {
        "full_sample": _slice_report(frame[TARGET_COLUMN], frame[GROUP_COLUMN], full_rates)
    }
    for partition in ("development", "test"):
        mask = splits["partition"].eq(partition)
        report[partition] = _slice_report(
            splits.loc[mask, TARGET_COLUMN],
            splits.loc[mask, GROUP_COLUMN],
            full_rates,
        )
        primary = mask & splits["primary_supervised_row"]
        report[f"{partition}_primary_rows"] = _slice_report(
            splits.loc[primary, TARGET_COLUMN],
            splits.loc[primary, GROUP_COLUMN],
            full_rates,
        )
    report["group_overlap"] = int(
        len(
            set(splits.loc[splits["partition"].eq("test"), GROUP_COLUMN])
            & set(splits.loc[splits["partition"].eq("development"), GROUP_COLUMN])
        )
    )
    report["test_group_fraction"] = report["test"]["n_groups"] / report["full_sample"]["n_groups"]
    report["test_row_fraction"] = report["test"]["n_rows"] / report["full_sample"]["n_rows"]
    return report


def _slice_report(labels: pd.Series, groups: pd.Series, full_rates: dict[str, float]) -> dict:
    counts = disposition_counts(labels)
    rates = disposition_rates(counts)
    return {
        "n_rows": int(len(labels)),
        "n_groups": int(groups.nunique()),
        "counts": counts,
        "rates": rates,
        "percentage_point_difference_from_full_sample": {
            name: 100.0 * (rates[name] - full_rates[name]) for name in CLASS_ORDER
        },
    }


def assert_split_integrity(splits: pd.DataFrame) -> None:
    test_groups = set(splits.loc[splits["partition"].eq("test"), GROUP_COLUMN])
    dev_groups = set(splits.loc[splits["partition"].eq("development"), GROUP_COLUMN])
    overlap = test_groups & dev_groups
    if overlap:
        raise AssertionError(f"{len(overlap)} host stars appear in both partitions.")
    if splits[ID_COLUMN].duplicated().any():
        raise AssertionError("A KOI name appears more than once in the split table.")
    if set(splits["partition"].unique()) != {"development", "test"}:
        raise AssertionError("Unexpected partition labels.")
    test_folds = splits.loc[splits["partition"].eq("test"), "dev_fold"]
    if not (test_folds == -1).all():
        raise AssertionError("Test rows were assigned a development CV fold.")
    dev = splits.loc[splits["partition"].eq("development")]
    if set(dev["dev_fold"].unique()) != set(range(DEV_CV_N_SPLITS)):
        raise AssertionError("Development CV folds do not cover 0..4.")
    fold_counts = dev.groupby(GROUP_COLUMN)["dev_fold"].nunique()
    if int((fold_counts > 1).sum()):
        raise AssertionError("A development host star crosses CV folds.")
    if dev["dev_fold"].lt(0).any():
        raise AssertionError("A development row has no CV fold.")
    for name in CLASS_ORDER:
        if not (splits.loc[splits["partition"].eq("test"), TARGET_COLUMN] == name).any():
            raise AssertionError(f"{name} is missing from the test partition.")
        if not (splits.loc[splits["partition"].eq("development"), TARGET_COLUMN] == name).any():
            raise AssertionError(f"{name} is missing from the development partition.")
