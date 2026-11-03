"""Write the Stage 1 findings from computed tables. Numbers are not hard-coded."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from exoplanet.config import (
    CLASS_ORDER,
    DATASET_DOI,
    DATASET_SOURCE,
    DATASET_VERSION_DATE,
    DICTIONARY_RETRIEVED,
    HOLDOUT_N_SPLITS,
    KOI_DICTIONARY_UPDATED,
    KOI_DICTIONARY_URL,
    MAX_RESIDUAL_MISSING_FRACTION,
    RANDOM_SEED,
    TCE_DICTIONARY_URL,
)


def write_findings(
    path: Path,
    record: dict,
    overview: dict,
    block: dict,
    audit: pd.DataFrame,
    balance: dict,
    imputed_columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    included = audit[audit["decision"] == "include"]
    excluded = audit[audit["decision"] == "exclude"]
    lines = [
        "# Stage 1 findings",
        "",
        "These counts were produced by `python -m exoplanet.stage1`. "
        "The feature list and split are a proposal for the team. They are not "
        "final until the other members agree. This stage does not train a "
        "Decision Tree or a Random Forest.",
        "",
        "## Dataset record",
        "",
        f"- Source: {DATASET_SOURCE}",
        f"- DOI: {DATASET_DOI}",
        f"- Version date in the filename: {DATASET_VERSION_DATE}",
        f"- Local file modification time: {record['local_mtime']}",
        f"- Canonical file: `{record['canonical_path']}`",
        f"- SHA-256: `{record['sha256']}`",
        f"- Bytes: {record['bytes']:,}",
        f"- Shape: {overview['n_rows']:,} rows × {overview['n_columns']} columns",
        f"- Duplicate rows: {overview['duplicate_rows']}",
        f"- Duplicate `kepoi_name`: {overview['duplicate_kepoi_name']}",
        f"- Host stars (`kepid`): {overview['n_kepid']:,}",
        f"- Host stars with more than one KOI: {overview['n_kepid_with_multiple_kois']:,} "
        f"({overview['n_kois_on_multi_koi_stars']:,} KOIs)",
        f"- Host stars with mixed dispositions: {overview['n_kepid_with_mixed_dispositions']:,}",
        f"- Column page: {KOI_DICTIONARY_URL}",
        f"- Column page last updated: {KOI_DICTIONARY_UPDATED}; retrieved {DICTIONARY_RETRIEVED}",
        f"- Random seed: {RANDOM_SEED}",
        "",
        "### Disposition counts",
        "",
        _count_table(overview["disposition_counts"], overview["disposition_rates"]),
        "",
        "FALSE POSITIVE is the largest class. Accuracy later has to be read against "
        "that base rate. This note is not a trained baseline model.",
        "",
        _anomaly_section(overview),
        "",
        "## Rows without a transit depth",
        "",
        _block_section(block),
        "",
        "## Feature decisions",
        "",
        f"Proposed scientific features: {len(included)}.",
        f"Excluded columns: {len(excluded)}.",
        "",
        "### Included features by family",
        "",
        _grouped_names(included, "feature_family"),
        "",
        "### Exclusion reasons",
        "",
        _exclusion_counts(excluded),
        "",
        "### Uncertainty columns",
        "",
        _uncertainty_section(audit),
        "",
        "### Centroid measurements and disposition flags",
        "",
        _vetting_section(audit),
        "",
        "## Proposed split",
        "",
        f"Seed {RANDOM_SEED}. `StratifiedGroupKFold` with {HOLDOUT_N_SPLITS} folds, "
        "shuffle on, holdout = fold 0. Development cross-validation is a separate "
        f"{HOLDOUT_N_SPLITS}-fold grouped split on the development stars only.",
        "",
        f"- Test host-star fraction: {balance['test_group_fraction']:.1%}",
        f"- Test row fraction: {balance['test_row_fraction']:.1%}",
        f"- Host stars in both partitions: {balance['group_overlap']}",
        "",
        _balance_section(balance),
        "",
        "## Preprocessing",
        "",
        "The proposed pipeline is a median imputer with no scaler. Trees do not "
        "need standardized features. The imputer is fit on development training "
        "rows only. Stage 1 does not transform the test partition.",
        "",
        "Rows without `koi_depth` stay in the split file and are omitted from the "
        "primary matrix, so their missing fitted parameters are not imputed.",
        "",
        f"Residual gaps are median-imputed only when they are at or below "
        f"{MAX_RESIDUAL_MISSING_FRACTION:.0%} of the rows that have a transit depth. "
        f"Columns with a residual gap: {len(imputed_columns)}.",
        "",
        _bullet(imputed_columns),
        "",
        "The broad diagnostic matrix adds `koi_pdisposition`, `koi_score`, the four "
        "`koi_fpflag` columns, and `has_kepler_name` on the same rows. It is not a "
        "scientific feature set, and its scores must not be reported as prediction "
        "performance.",
        "",
        "## Proposal status",
        "",
        "Pending agreement from Sanjana Reddy Katham, Nathan Pak, and "
        "Venkata Sai Bhavana Yarrapragada. Person 3 prepared this configuration "
        "so the team can share one split and one leakage-aware column list.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _anomaly_section(overview: dict) -> str:
    lines = ["### Values that are not 0/1 false-positive flags", ""]
    anomalies = overview.get("fpflag_values_other_than_0_or_1", [])
    if not anomalies:
        lines.append("Every false-positive flag is 0 or 1.")
    else:
        lines.append(
            "These values are left unchanged in the raw file. The flags are "
            "excluded from the scientific feature set. The diagnostic matrix "
            "keeps the stored value."
        )
        lines.append("")
        for item in anomalies:
            lines.append(
                f"- `{item['kepoi_name']}` has `{item['column']}` = {item['value']:g}."
            )
    lines.extend(["", "### Repeated numeric values checked as possible sentinels", ""])
    sentinels = overview.get("sentinel_values", [])
    if not sentinels:
        lines.append("No column contained 20 or more exact values of 99 or -99.")
    else:
        lines.append(
            "The column page notes that older deliveries sometimes used 99 or -99 "
            "for processing problems and that those codes were later stored as nulls. "
            "A stellar-temperature uncertainty of 99 K is also a plausible measurement, "
            "so these cells were not recoded."
        )
        lines.append("")
        for item in sentinels:
            lines.append(
                f"- `{item['column']}` equals {item['value']} on {item['count']} rows."
            )
    return "\n".join(lines)


def _count_table(counts: dict, rates: dict) -> str:
    rows = [
        "| Disposition | Count | Share |",
        "| --- | ---: | ---: |",
    ]
    for name in CLASS_ORDER:
        rows.append(f"| {name} | {counts[name]:,} | {rates[name]:.1%} |")
    return "\n".join(rows)


def _block_section(block: dict) -> str:
    unfitted = block["unfitted_disposition_rates"]["FALSE POSITIVE"]
    other = block["other_disposition_rates"]["FALSE POSITIVE"]
    observed = block["columns_fully_observed_on_unfitted_rows"]
    missing = block["columns_always_missing_on_unfitted_rows"]
    lines = [
        f"- `koi_fittype` = none: {block['n_fittype_none']:,} rows",
        f"- Rows with missing `koi_depth`: {block['n_unfitted_rows']:,}",
        f"- Fit type none but depth present: {block['n_fittype_none_with_depth']:,}",
        f"- Depth missing and fit type not none: {block['n_depth_missing_but_fittype_not_none']:,}",
        "",
        "The column page says fit type none means a fit is not provided and only "
        "orbital period, transit epoch, and transit duration are reported. "
        f"In this file, {block['n_fittype_none_with_depth']} of the "
        f"{block['n_fittype_none']} rows labeled none still have a transit depth "
        "and the associated fitted parameters, so those rows stay in the primary "
        "matrix. Every row with missing depth has fit type none. "
        f"FALSE POSITIVE is {unfitted:.1%} of rows with no transit depth and "
        f"{other:.1%} of rows with a transit depth.",
        "",
        "Median imputation is not appropriate for this block. It would fill planet "
        "and stellar parameters the archive did not measure. Leaving the cells "
        "missing is also not neutral: scikit-learn 1.9 trees can split on a missing "
        "value, and here the missingness is associated with disposition. The "
        "primary experiment therefore uses only rows with a measured transit depth. "
        "The rows without depth are not deleted. Each keeps the development or "
        "test assignment already stored in `data/processed/proposed_splits.csv`, "
        "and those assignments are also listed in "
        "`data/processed/incomplete_koi_split_assignments.csv` for a later "
        "sensitivity experiment.",
        "",
        "### Disposition of rows with no transit depth",
        "",
        _count_table(block["unfitted_disposition_counts"], block["unfitted_disposition_rates"]),
        "",
        "### Disposition of rows with a transit depth",
        "",
        _count_table(block["other_disposition_counts"], block["other_disposition_rates"]),
        "",
        f"### Columns missing on every row with no transit depth ({len(missing)})",
        "",
        _bullet(missing),
        "",
        f"### Columns observed on every row with no transit depth ({len(observed)})",
        "",
        _bullet(observed),
        "",
        "Columns missing for every row that lacks a transit depth, and missing on "
        f"less than {5}% of rows that have a depth:",
        "",
        _bullet(block["co_missing_on_depth_gap"]),
    ]
    return "\n".join(lines)


def _grouped_names(frame: pd.DataFrame, column: str) -> str:
    lines = []
    for family, part in frame.groupby(column, sort=True):
        names = ", ".join(f"`{name}`" for name in part["column"].tolist())
        lines.append(f"- **{family}** ({len(part)}): {names}")
    return "\n".join(lines) if lines else "- None"


def _exclusion_counts(excluded: pd.DataFrame) -> str:
    counts = excluded["exclusion_group"].value_counts()
    rows = ["| Reason | Columns |", "| --- | ---: |"]
    for name, count in counts.items():
        rows.append(f"| {name} | {int(count)} |")
    return "\n".join(rows)


def _uncertainty_section(audit: pd.DataFrame) -> str:
    errors = audit[audit["kind"] == "uncertainty"].copy()
    if errors.empty:
        return "No uncertainty columns were found."
    counts = errors["decision"].value_counts()
    lines = [
        "Each uncertainty column was reviewed on its own. Empty columns, columns "
        "whose parent measurement was excluded, exact negative mirrors of `err1`, "
        "and delivery-wide gaps are excluded. The others are proposed features.",
        "",
        f"- Uncertainty columns in the CSV: {len(errors)}",
        f"- Proposed for inclusion: {int(counts.get('include', 0))}",
        f"- Excluded: {int(counts.get('exclude', 0))}",
        "",
        "| Column | Parent | Missing on depth-measured rows | Decision | Why excluded |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for _, row in errors.sort_values(["decision", "column"]).iterrows():
        missing = f"{row['missing_fraction_primary_rows']:.1%}"
        why = "" if row["decision"] == "include" else row["exclusion_group"]
        parent = row["parent"] or ""
        lines.append(
            f"| `{row['column']}` | `{parent}` | {missing} | {row['decision']} | {why} |"
        )
    lines.append("")
    lines.append(
        "Full sentences for each decision are in `results/shared/column_audit.csv`."
    )
    return "\n".join(lines)


def _vetting_section(audit: pd.DataFrame) -> str:
    names = [
        "koi_fpflag_nt",
        "koi_fpflag_ss",
        "koi_fpflag_co",
        "koi_fpflag_ec",
        "koi_score",
        "koi_pdisposition",
        "koi_fwm_sra",
        "koi_fwm_sdec",
        "koi_fwm_srao",
        "koi_fwm_sdeco",
        "koi_fwm_prao",
        "koi_fwm_pdeco",
        "koi_fwm_stat_sig",
        "koi_dicco_mra",
        "koi_dicco_mdec",
        "koi_dicco_msky",
        "koi_dikco_mra",
        "koi_dikco_mdec",
        "koi_dikco_msky",
        "koi_bin_oedp_sig",
    ]
    lines = [
        "The column page separates pixel measurements from the centroid-offset "
        "flag. `koi_fpflag_co` is the flag inferred from centroid measurements. "
        "The offset columns are measurements from flux-weighted centroids or "
        "PRF-fit difference images, not copies of `koi_disposition`. They stay "
        "provisional because they come from the vetting analysis and may be "
        "strong predictors. `koi_fwm_stat_sig` is not an offset. The column page "
        "defines it as a significance percent, where 100% means confidence that "
        "the transit is on the target star, so it is removed from the primary "
        "set and saved in `experimental_features_with_fwm_stat_sig.csv`. "
        "Absolute source coordinates (`koi_fwm_sra`, `koi_fwm_sdec`) are "
        "excluded as sky positions.",
        "",
        "| Column | Decision | Provisional | Group |",
        "| --- | --- | --- | --- |",
    ]
    indexed = audit.set_index("column")
    for name in names:
        if name not in indexed.index:
            lines.append(f"| `{name}` | not in table |  |  |")
            continue
        row = indexed.loc[name]
        lines.append(
            f"| `{name}` | {row['decision']} | {bool(row['provisional'])} | {row['exclusion_group']} |"
        )
    return "\n".join(lines)


def _balance_section(balance: dict) -> str:
    lines = [
        "| Sample | Rows | Host stars | CONFIRMED | CANDIDATE | FALSE POSITIVE | Largest shift (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in (
        "full_sample",
        "development",
        "test",
        "development_primary_rows",
        "test_primary_rows",
    ):
        part = balance[key]
        rates = part["rates"]
        shifts = part["percentage_point_difference_from_full_sample"]
        largest = max(abs(value) for value in shifts.values())
        lines.append(
            f"| {key} | {part['n_rows']:,} | {part['n_groups']:,} | "
            f"{rates['CONFIRMED']:.1%} | {rates['CANDIDATE']:.1%} | "
            f"{rates['FALSE POSITIVE']:.1%} | {largest:.2f} |"
        )
    lines.append("")
    n_rows = balance["full_sample"]["n_rows"]
    lines.append(
        f"Shifts are percentage points relative to the full table of {n_rows:,} rows."
    )
    return "\n".join(lines)


def _bullet(items: list[str]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- `{item}`" for item in items)
