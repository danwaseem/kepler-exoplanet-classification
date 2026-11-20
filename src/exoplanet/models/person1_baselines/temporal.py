"""E5 temporal validation across two versions of the KOI cumulative table.

The older Kaggle copy of the table and the NASA 2026-10-06 table share
kepoi_name and kepid. Between the two versions, some KOIs that were
CANDIDATE have been resolved to CONFIRMED or FALSE POSITIVE. That gives a
forward-in-time test: train only on what the old catalog said, then predict
how its open candidates were later resolved.

Rules
-----
* Only host stars in the development partition of the locked split are used.
  Stars in the test partition are dropped before anything else.
* Features are the old catalog values, restricted to columns that exist in
  both the old file and the 76 proposed features.
* Rows need a measured koi_depth in the old file (same rule as the primary
  experiment).
* Training stars and evaluation stars are disjoint: any training row whose
  kepid hosts an evaluation object is dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from exoplanet.config import DATA_PROCESSED, GROUP_COLUMN, ID_COLUMN, RAW_CSV, ROOT, TARGET_COLUMN
from exoplanet.io import file_sha256, load_koi

KAGGLE_CSV = ROOT / "experiments" / "person1_baselines" / "data" / "kaggle_cumulative.csv"
CLASSES = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]


def load_versions(kaggle_path: Path = KAGGLE_CSV) -> pd.DataFrame:
    """Old rows joined to their current label and split partition."""
    old = pd.read_csv(kaggle_path, low_memory=False)
    new = load_koi(RAW_CSV)[[ID_COLUMN, GROUP_COLUMN, TARGET_COLUMN]]
    if old[ID_COLUMN].duplicated().any() or new[ID_COLUMN].duplicated().any():
        raise RuntimeError("kepoi_name is not unique.")
    merged = old.merge(
        new.rename(columns={TARGET_COLUMN: "label_new", GROUP_COLUMN: "kepid_new"}),
        on=ID_COLUMN, how="left", validate="one_to_one",
    ).rename(columns={TARGET_COLUMN: "label_old"})
    if merged["label_new"].isna().any():
        raise RuntimeError("Some old KOIs are missing from the current table.")
    if (merged[GROUP_COLUMN] != merged["kepid_new"]).any():
        raise RuntimeError("kepid changed between versions for some KOIs.")
    splits = pd.read_csv(DATA_PROCESSED / "proposed_splits.csv")
    star_partition = splits.drop_duplicates(GROUP_COLUMN).set_index(GROUP_COLUMN)["partition"]
    if splits.groupby(GROUP_COLUMN)["partition"].nunique().max() != 1:
        raise RuntimeError("A host star is in more than one partition.")
    merged["partition"] = merged[GROUP_COLUMN].map(star_partition)
    return merged


def label_crosstab(merged: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(merged["label_old"], merged["label_new"])
    table = table.reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    table.index.name = "old label (Kaggle copy)"
    table.columns.name = "current label (NASA 2026-10-06)"
    return table


def crosstab_summary(table: pd.DataFrame) -> dict:
    total = int(table.to_numpy().sum())
    same = int(sum(table.loc[c, c] for c in CLASSES))
    return {
        "matched_kois": total,
        "label_changed": total - same,
        "old_candidate_resolved": int(table.loc["CANDIDATE", "CONFIRMED"]
                                      + table.loc["CANDIDATE", "FALSE POSITIVE"]),
        "old_candidate_to_confirmed": int(table.loc["CANDIDATE", "CONFIRMED"]),
        "old_candidate_to_false_positive": int(table.loc["CANDIDATE", "FALSE POSITIVE"]),
        "old_candidate_still_candidate": int(table.loc["CANDIDATE", "CANDIDATE"]),
        "old_false_positive_to_candidate": int(table.loc["FALSE POSITIVE", "CANDIDATE"]),
        "old_false_positive_to_confirmed": int(table.loc["FALSE POSITIVE", "CONFIRMED"]),
        "old_confirmed_to_false_positive": int(table.loc["CONFIRMED", "FALSE POSITIVE"]),
        "old_confirmed_to_candidate": int(table.loc["CONFIRMED", "CANDIDATE"]),
    }


def temporal_sets(merged: pd.DataFrame, primary_features: list[str]) -> dict:
    """Training and evaluation sets for E5, all from the development partition."""
    features = [c for c in primary_features if c in merged.columns]
    lost = [c for c in primary_features if c not in merged.columns]
    counts: dict[str, int] = {"old_rows": int(len(merged))}
    dev = merged.loc[merged["partition"].eq("development")].copy()
    counts["test_partition_rows_dropped"] = int(merged["partition"].eq("test").sum())
    counts["rows_without_partition_dropped"] = int(merged["partition"].isna().sum())
    counts["development_rows"] = int(len(dev))
    has_depth = dev["koi_depth"].notna()
    counts["development_rows_missing_old_depth_dropped"] = int((~has_depth).sum())
    dev = dev.loc[has_depth]
    for c in features:
        dev[c] = pd.to_numeric(dev[c], errors="coerce")

    evaluation = dev.loc[
        dev["label_old"].eq("CANDIDATE") & dev["label_new"].isin(["CONFIRMED", "FALSE POSITIVE"])
    ].copy()
    eval_stars = set(evaluation[GROUP_COLUMN])
    pool = dev.loc[dev["label_old"].isin(["CONFIRMED", "FALSE POSITIVE"])]
    shares_star = pool[GROUP_COLUMN].isin(eval_stars)
    counts["training_rows_before_star_exclusion"] = int(len(pool))
    counts["training_rows_dropped_shared_star"] = int(shares_star.sum())
    train = pool.loc[~shares_star].copy()
    changed = train["label_old"].ne(train["label_new"])
    counts["training_rows"] = int(len(train))
    counts["training_rows_label_changed_since"] = int(changed.sum())
    counts["evaluation_rows"] = int(len(evaluation))
    counts["evaluation_now_confirmed"] = int(evaluation["label_new"].eq("CONFIRMED").sum())
    counts["evaluation_now_false_positive"] = int(evaluation["label_new"].eq("FALSE POSITIVE").sum())
    counts["training_old_confirmed"] = int(train["label_old"].eq("CONFIRMED").sum())
    counts["training_old_false_positive"] = int(train["label_old"].eq("FALSE POSITIVE").sum())
    if set(train[GROUP_COLUMN]) & eval_stars:
        raise RuntimeError("A training star hosts an evaluation object.")
    if not dev["partition"].eq("development").all():
        raise RuntimeError("A non-development row entered E5.")
    return {
        "features": features,
        "lost_features": lost,
        "train": train,
        "train_label_stable": train.loc[~changed].copy(),
        "evaluation": evaluation,
        "counts": counts,
        "kaggle_sha256": file_sha256(KAGGLE_CSV),
    }
