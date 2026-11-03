"""Training-fold imputation for residual gaps. The unfitted block is not filled."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from exoplanet.config import ID_COLUMN, TARGET_COLUMN


def make_imputer() -> Pipeline:
    """Median imputer with no scaler.

    Decision trees and random forests do not need standardized inputs.
    The imputer is for residual gaps on rows that already have a transit
    depth. It must be fit on development training rows only.
    """
    return Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(strategy="median", keep_empty_features=True),
            )
        ]
    )


def primary_learning_frame(
    frame: pd.DataFrame,
    splits: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Unimputed scientific matrix for rows that have a measured transit depth.

    The returned frame keeps the source row index. Identifier, label, and
    partition columns are included so later training can filter them. They are
    not model features. Nothing is fit here, and residual gaps stay missing.
    """
    if not splits.index.equals(frame.index):
        raise ValueError("Split table must be aligned to the KOI table.")
    kept = splits["primary_supervised_row"]
    base = splits.loc[kept, [ID_COLUMN, "kepid", TARGET_COLUMN, "partition", "dev_fold"]]
    features = frame.loc[kept, feature_columns].apply(pd.to_numeric, errors="coerce")
    return pd.concat([base, features], axis=1)


def diagnostic_learning_frame(
    frame: pd.DataFrame,
    primary: pd.DataFrame,
) -> pd.DataFrame:
    """Same rows as the scientific matrix, with disposition fields added.

    This extract is a leakage diagnostic. It is not the scientific feature set.
    has_kepler_name is 1 when a Kepler name is present. The column page says
    that name identifies a confirmed or validated planet.
    """
    extra_columns = [
        "koi_pdisposition",
        "koi_score",
        "koi_fpflag_nt",
        "koi_fpflag_ss",
        "koi_fpflag_co",
        "koi_fpflag_ec",
    ]
    extras = frame.loc[primary.index, extra_columns].copy()
    for column in extra_columns[1:]:
        extras[column] = pd.to_numeric(extras[column], errors="coerce")
    name = frame.loc[primary.index, "kepler_name"]
    extras["has_kepler_name"] = name.notna() & name.astype(str).str.strip().ne("")
    extras["has_kepler_name"] = extras["has_kepler_name"].astype(int)
    return pd.concat([primary, extras], axis=1)


def fit_imputer(
    features: pd.DataFrame,
    partition: pd.Series,
    train_index: pd.Index,
) -> Pipeline:
    """Fit the residual-gap imputer. Refuses any index outside development."""
    seen = set(partition.loc[train_index].unique())
    if seen != {"development"}:
        raise ValueError(
            "Refusing to fit preprocessing on rows outside the development set. "
            f"Saw partitions: {sorted(seen)}."
        )
    pipeline = make_imputer()
    pipeline.fit(features.loc[train_index])
    return pipeline


def training_fold_index(splits: pd.DataFrame, fold_id: int) -> pd.Index:
    """Development rows whose CV fold is not the validation fold."""
    mask = (
        splits["partition"].eq("development")
        & splits["primary_supervised_row"]
        & splits["dev_fold"].ne(fold_id)
    )
    return splits.index[mask]


def assert_imputer_uses_training_rows(
    features: pd.DataFrame,
    pipeline: Pipeline,
    train_index: pd.Index,
) -> None:
    """Median statistics must match the training rows and ignore other rows."""
    imputer: SimpleImputer = pipeline.named_steps["impute"]
    expected = np.nanmedian(features.loc[train_index].to_numpy(dtype=float), axis=0)
    actual = np.asarray(imputer.statistics_, dtype=float)
    if not np.allclose(actual, expected, equal_nan=True):
        raise AssertionError("Imputer medians do not match the training fold.")
