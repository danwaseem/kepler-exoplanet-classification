"""Leakage audit, unfitted-row analysis, and the proposed feature decision."""

from __future__ import annotations

import numpy as np
import pandas as pd

from exoplanet.config import (
    CLASS_ORDER,
    FREE_FIT_PARAMETERS,
    GROUP_COLUMN,
    ID_COLUMN,
    INSOL_ERROR_COLUMNS,
    LEAKAGE_COLUMNS,
    LIMB_DARKENING_COLUMNS,
    MAX_RESIDUAL_MISSING_FRACTION,
    METADATA_COLUMNS,
    MIN_PAIRS_FOR_SYMMETRY,
    ODD_EVEN_COLUMN,
    PIXEL_OFFSET_COLUMNS,
    SKY_POSITION_COLUMNS,
    STRUCTURAL_GROUP_MIN_ROWS,
    STRUCTURAL_PRESENT_MAX_MISSING,
    SYMMETRIC_ATOL,
    SYMMETRIC_ERROR_FRACTION,
    SYMMETRIC_RTOL,
    TARGET_COLUMN,
    UNFITTED_BLOCK_OTHER_MISSING_MAX,
)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _missing_fraction(series: pd.Series) -> float:
    return float(series.isna().mean()) if len(series) else 0.0


def disposition_counts(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts()
    return {name: int(counts.get(name, 0)) for name in CLASS_ORDER}


def disposition_rates(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {name: 0.0 for name in counts}
    return {name: counts[name] / total for name in counts}


def profile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        numeric = _numeric(frame[column])
        observed = numeric.notna().sum()
        rows.append(
            {
                "column": column,
                "dtype": str(frame[column].dtype),
                "n_missing": int(frame[column].isna().sum()),
                "missing_fraction": _missing_fraction(frame[column]),
                "n_unique_non_missing": int(frame[column].nunique(dropna=True)),
                "n_numeric": int(observed),
                "numeric_fraction_of_observed": (
                    float(observed / frame[column].notna().sum())
                    if frame[column].notna().any()
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def dataset_overview(frame: pd.DataFrame) -> dict:
    sizes = frame.groupby(GROUP_COLUMN).size()
    mixed = frame.groupby(GROUP_COLUMN)[TARGET_COLUMN].nunique()
    counts = disposition_counts(frame[TARGET_COLUMN])
    anomalies = []
    for column in [name for name in LEAKAGE_COLUMNS if name.startswith("koi_fpflag")]:
        if column not in frame.columns:
            continue
        values = _numeric(frame[column])
        bad = frame.loc[~values.isin([0, 1]) & values.notna(), ID_COLUMN]
        for name in bad.astype(str):
            anomalies.append(
                {
                    "column": column,
                    "kepoi_name": name,
                    "value": float(values.loc[frame[ID_COLUMN].astype(str) == name].iloc[0]),
                }
            )
    sentinels = []
    for column in frame.columns:
        numeric = _numeric(frame[column])
        for sentinel in (-99, 99):
            count = int((numeric == sentinel).sum())
            if count >= 20:
                sentinels.append({"column": column, "value": sentinel, "count": count})
    return {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_kepoi_name": int(frame[ID_COLUMN].duplicated().sum()),
        "n_kepid": int(sizes.shape[0]),
        "n_kepid_with_multiple_kois": int((sizes > 1).sum()),
        "n_kois_on_multi_koi_stars": int(sizes[sizes > 1].sum()),
        "n_kepid_with_mixed_dispositions": int((mixed > 1).sum()),
        "disposition_counts": counts,
        "disposition_rates": disposition_rates(counts),
        "fpflag_values_other_than_0_or_1": anomalies,
        "sentinel_values": sentinels,
    }


def _fittype_none(frame: pd.DataFrame) -> pd.Series:
    return frame["koi_fittype"].fillna("").astype(str).str.strip().eq("none")


def identify_unfitted_block(frame: pd.DataFrame) -> dict:
    """Describe rows that have no measured transit depth.

    The column page says fit type "none" means a fit is not provided and only
    period, epoch, and duration are reported. In this file that label is a
    little broader than the missing-depth block, so the row rule follows the
    measured gap: a row is unfitted when koi_depth is missing. Columns that
    are missing for every such row, and nearly complete otherwise, are the
    co-missing block. Rows that still have a depth are kept even if their
    fit-type label is none.
    """
    none = _fittype_none(frame)
    if "koi_depth" not in frame.columns:
        raise RuntimeError("koi_depth is not in the table.")
    unfitted = _numeric(frame["koi_depth"]).isna()
    if not unfitted.any():
        raise RuntimeError("No rows are missing koi_depth; the block rule needs review.")

    co_missing = []
    for column in frame.columns:
        numeric = _numeric(frame[column])
        if int(numeric.notna().sum()) < 100:
            continue
        missing_on_unfitted = _missing_fraction(numeric.loc[unfitted])
        missing_on_other = _missing_fraction(numeric.loc[~unfitted])
        if missing_on_unfitted == 1.0 and missing_on_other < UNFITTED_BLOCK_OTHER_MISSING_MAX:
            co_missing.append(column)
    if "koi_depth" not in co_missing:
        raise RuntimeError(
            "koi_depth is not jointly missing with the rest of the unfitted block."
        )

    present_on_unfitted = []
    always_missing_on_unfitted = []
    for column in frame.columns:
        fraction = _missing_fraction(frame.loc[unfitted, column])
        if fraction == 0.0:
            present_on_unfitted.append(column)
        elif fraction == 1.0:
            always_missing_on_unfitted.append(column)

    none_and_depth = none & ~unfitted
    depth_without_none = unfitted & ~none
    unfitted_counts = disposition_counts(frame.loc[unfitted, TARGET_COLUMN])
    other_counts = disposition_counts(frame.loc[~unfitted, TARGET_COLUMN])
    return {
        "n_fittype_none": int(none.sum()),
        "n_unfitted_rows": int(unfitted.sum()),
        "n_fittype_none_with_depth": int(none_and_depth.sum()),
        "n_depth_missing_but_fittype_not_none": int(depth_without_none.sum()),
        "co_missing_on_depth_gap": co_missing,
        "columns_always_missing_on_unfitted_rows": always_missing_on_unfitted,
        "columns_fully_observed_on_unfitted_rows": present_on_unfitted,
        "unfitted_disposition_counts": unfitted_counts,
        "other_disposition_counts": other_counts,
        "unfitted_disposition_rates": disposition_rates(unfitted_counts),
        "other_disposition_rates": disposition_rates(other_counts),
        "row_mask": unfitted,
    }


def _koi_count_assessment(frame: pd.DataFrame) -> dict:
    """Compare koi_count with row multiplicity and with non-false-positive counts.

    The column page describes this field as the number of planet candidates
    in a system. If the stored values instead match the number of KOI rows,
    the field is system multiplicity rather than a count of the target class.
    """
    n_rows = frame.groupby(GROUP_COLUMN).size()
    stored = _numeric(frame.groupby(GROUP_COLUMN)["koi_count"].first())
    not_fp = (
        frame.loc[frame[TARGET_COLUMN].isin(["CONFIRMED", "CANDIDATE"])]
        .groupby(GROUP_COLUMN)
        .size()
        .reindex(n_rows.index, fill_value=0)
    )
    return {
        "fraction_equal_to_koi_rows": float((stored == n_rows).mean()),
        "fraction_equal_to_confirmed_or_candidate": float((stored == not_fp).mean()),
        "dictionary_label": "Number of Planets",
    }


def _epoch_duplicate(frame: pd.DataFrame) -> dict:
    """koi_time0 is the BK epoch plus the archive's 2,454,833 day offset.

    Stored values can differ from that constant by a fraction of a minute
    because the two columns are rounded separately. Differences no larger
    than 0.001 day are treated as the same ephemeris, not a second measurement.
    """
    if "koi_time0" not in frame.columns or "koi_time0bk" not in frame.columns:
        return {"is_duplicate": False, "offset": None, "max_abs_days": None, "n_compared": 0}
    offset = _numeric(frame["koi_time0"]) - _numeric(frame["koi_time0bk"])
    observed = offset.dropna()
    if observed.empty:
        return {"is_duplicate": False, "offset": None, "max_abs_days": None, "n_compared": 0}
    defined_offset = 2454833.0
    max_abs_days = float((observed - defined_offset).abs().max())
    median_offset = float(observed.median())
    return {
        "is_duplicate": bool(max_abs_days <= 1e-3 and abs(median_offset - defined_offset) <= 1e-3),
        "offset": median_offset,
        "max_abs_days": max_abs_days,
        "n_compared": int(observed.shape[0]),
    }


def _structural_delivery_gap(frame: pd.DataFrame, column: str, primary: pd.Series) -> bool:
    """True when one large parameter delivery lacks the column and another has it."""
    subset = frame.loc[primary, ["koi_parm_prov", column]].copy()
    subset["_missing"] = subset[column].isna()
    summary = subset.groupby("koi_parm_prov", dropna=False)["_missing"].agg(
        mean="mean", size="size"
    )
    fully_missing = summary[
        (summary["mean"] == 1.0) & (summary["size"] >= STRUCTURAL_GROUP_MIN_ROWS)
    ]
    present = summary[
        (summary["mean"] <= STRUCTURAL_PRESENT_MAX_MISSING)
        & (summary["size"] >= STRUCTURAL_GROUP_MIN_ROWS)
    ]
    return bool(len(fully_missing) and len(present))


def _symmetry_fraction(left: pd.Series, right: pd.Series, primary: pd.Series) -> float | None:
    both = primary & left.notna() & right.notna()
    if int(both.sum()) < MIN_PAIRS_FOR_SYMMETRY:
        return None
    return float(
        np.isclose(
            left.loc[both].to_numpy(),
            -right.loc[both].to_numpy(),
            rtol=SYMMETRIC_RTOL,
            atol=SYMMETRIC_ATOL,
        ).mean()
    )


def _family(column: str, kind: str, section: str) -> str:
    if kind == "uncertainty":
        return "uncertainty"
    if column in PIXEL_OFFSET_COLUMNS or column == ODD_EVEN_COLUMN:
        return "vetting_measurement"
    if column in FREE_FIT_PARAMETERS:
        return "transit_fit"
    if column in {"koi_steff", "koi_slogg", "koi_smet", "koi_srad", "koi_smass", "koi_sage"}:
        return "stellar"
    if section.startswith("KIC") or column.startswith("koi_") and column.endswith("mag"):
        return "photometry"
    if column in {"koi_model_snr", "koi_max_sngle_ev", "koi_max_mult_ev", "koi_num_transits"}:
        return "tce_statistic"
    if column == "koi_count":
        return "multiplicity"
    if "Transit" in section:
        return "model_derived_transit"
    if "Stellar" in section:
        return "stellar"
    if "Pixel" in section:
        return "vetting_measurement"
    return "other"


def build_feature_audit(
    frame: pd.DataFrame,
    documents: dict[str, dict],
    odd_even_doc: dict,
    block: dict,
) -> pd.DataFrame:
    """Decide each CSV column from the dictionary plus measured availability."""
    documents = dict(documents)
    documents[ODD_EVEN_COLUMN] = odd_even_doc
    for err_name, parent in INSOL_ERROR_COLUMNS.items():
        parent_doc = documents.get(parent, {})
        documents[err_name] = {
            "column": err_name,
            "section": parent_doc.get("section", "Transit Properties"),
            "label": "Uncertainty on Insolation Flux [Earth flux]",
            "description": (
                "The KOI column page leaves the uncertainty cell for koi_insol "
                "blank. The CSV still contains this err companion. It is reviewed "
                "as an insolation uncertainty, with that documentation gap recorded."
            ),
            "kind": "uncertainty",
            "parent": parent,
            "uncertainty_columns": [],
            "source_file": parent_doc.get("source_file", ""),
        }

    primary = ~block["row_mask"]
    numeric = {column: _numeric(frame[column]) for column in frame.columns}
    count_info = _koi_count_assessment(frame) if "koi_count" in frame.columns else {}
    epoch_info = _epoch_duplicate(frame)
    uncertainty_of = {
        name: doc.get("parent", "")
        for name, doc in documents.items()
        if doc.get("kind") == "uncertainty" and doc.get("parent")
    }

    records = []
    for column in frame.columns:
        doc = documents.get(column, {})
        kind = doc.get("kind", "")
        section = doc.get("section", "")
        label = doc.get("label", "")
        description = doc.get("description", "")
        values = numeric[column]
        missing_all = _missing_fraction(frame[column])
        missing_primary = _missing_fraction(frame.loc[primary, column])
        n_unique = int(frame[column].nunique(dropna=True))
        numeric_observed = int(values.notna().sum())
        is_numeric = numeric_observed >= 100 or (
            frame[column].notna().any()
            and numeric_observed == int(frame[column].notna().sum())
        )
        decision = "include"
        group = "included"
        reason = ""
        provisional = False
        impute = False
        parent = uncertainty_of.get(column, doc.get("parent", ""))

        if column == TARGET_COLUMN:
            decision, group = "exclude", "target"
            reason = (
                "Exoplanet Archive Disposition is the supervised target. "
                "Current values include CANDIDATE, FALSE POSITIVE, and CONFIRMED."
            )
        elif column in LEAKAGE_COLUMNS:
            decision, group = "exclude", "disposition_leakage"
            reason = _leakage_reason(column, label, description)
        elif column in METADATA_COLUMNS:
            decision, group = "exclude", "metadata_or_identifier"
            reason = _metadata_reason(column, label, n_unique)
        elif column in SKY_POSITION_COLUMNS:
            decision, group = "exclude", "sky_position"
            reason = (
                f"{label or column} is a sky coordinate. It is shared by every KOI "
                "of a host star and is not a transit, stellar, or centroid-offset "
                "measurement. Absolute flux-weighted source positions are excluded "
                "on the same basis. Centroid offsets are reviewed separately."
            )
        elif column in LIMB_DARKENING_COLUMNS:
            decision, group = "exclude", "fixed_model_input"
            reason = (
                "The column page says limb-darkening coefficients depend on stellar "
                "temperature, surface gravity, and metallicity, are required inputs "
                "to the Mandel-Agol fit, and remain fixed during fit minimization. "
                "They are not independent measurements."
            )
        elif epoch_info.get("is_duplicate") and column in {
            "koi_time0",
            "koi_time0_err1",
            "koi_time0_err2",
        }:
            decision, group = "exclude", "duplicate_measurement"
            reason = (
                f"koi_time0 differs from koi_time0bk by the 2,454,833 day offset "
                f"defined on the column page. Across {epoch_info['n_compared']} rows "
                f"the largest absolute departure from that offset is "
                f"{epoch_info['max_abs_days']:.6f} days, consistent with separate "
                "rounding of the same epoch. The BK epoch is retained. Its "
                "uncertainties match the koi_time0 uncertainties exactly."
            )
        elif column == "koi_count":
            decision, group, reason = _count_decision(count_info)
        elif column == ODD_EVEN_COLUMN:
            provisional = True
            reason = (
                "Not listed on the KOI column page. The TCE page defines "
                "tce_bin_oedp_sig as the odd-even depth comparison and says a larger "
                "value makes an astrophysical false positive more likely. It is a "
                "light-curve statistic, not the archive disposition. "
            )
        elif column not in documents:
            decision, group = "exclude", "undocumented"
            reason = (
                "This CSV column is not defined on the KOI cumulative column page "
                "retrieved 2026-10-31, and it is not one of the reviewed companions "
                "(insolation uncertainties or koi_bin_oedp_sig)."
            )
        elif missing_all == 1.0 or numeric_observed == 0:
            decision, group = "exclude", "unavailable"
            reason = "The column is empty in this cumulative table."
        elif n_unique <= 1:
            decision, group = "exclude", "no_variation"
            observed = frame[column].dropna()
            example = observed.iloc[0].item() if hasattr(observed.iloc[0], "item") else observed.iloc[0]
            reason = (
                f"Every non-missing value is {example}. The column has no "
                "variation that a tree can split on."
            )
        elif not is_numeric:
            decision, group = "exclude", "non_numeric"
            reason = "The column is not a numeric measurement."
        elif kind == "uncertainty":
            reason = (
                f"Uncertainty column for {parent or 'a parent measurement'}, "
                "as named on the archive column page."
            )
            if column in INSOL_ERROR_COLUMNS:
                reason = (
                    "Reviewed individually as an insolation uncertainty. The KOI "
                    "column page leaves the koi_insol uncertainty cell blank; the "
                    "CSV still contains this companion column."
                )
        elif column == "koi_fwm_stat_sig":
            decision, group = "exclude", "vetting_summary"
            reason = (
                "The column page defines flux-weighted offset significance as a "
                "percent: 100% means there is no offset and there is confidence "
                "that the transit is on the target star. That is a derived "
                "summary of the centroid test, not the offset measurement. It is "
                "removed from the primary leakage-aware feature set and kept in "
                "a separate experimental configuration."
            )
        elif column in PIXEL_OFFSET_COLUMNS:
            provisional = True
            reason = (
                "Pixel-based vetting statistic computed from flux-weighted centroids "
                "or PRF-fit difference images. The column page describes these as "
                "measurements from target pixel files. They are not koi_disposition "
                "and they are not koi_fpflag_co, which is the separate flag inferred "
                "from centroid measurements. Kept as a provisional candidate feature."
            )
        else:
            reason = (
                f"Included from the column page ({section or 'undocumented section'}"
                f"{': ' + label if label else ''}). "
                + (
                    "The column-page introduction lists this among the free parameters of the transit fit."
                    if column in FREE_FIT_PARAMETERS
                    else "Treated as a modeled or catalog measurement rather than a disposition product."
                )
            )

        records.append(
            {
                "column": column,
                "decision": decision,
                "exclusion_group": group,
                "provisional": provisional,
                "impute_residual": impute,
                "feature_family": _family(column, kind or ("uncertainty" if parent else ""), section),
                "dictionary_section": section,
                "dictionary_label": label,
                "parent": parent,
                "kind": kind,
                "missing_fraction_all_rows": missing_all,
                "missing_fraction_primary_rows": missing_primary,
                "n_unique_non_missing": n_unique,
                "symmetric_negative_fraction": np.nan,
                "reason": " ".join(reason.split()),
                "description": description,
            }
        )

    audit = pd.DataFrame(records).set_index("column", drop=False)

    # Uncertainties inherit an excluded parent, then availability is applied
    # only to columns that are still candidates.
    for column, row in audit.iterrows():
        parent = row["parent"]
        if row["kind"] != "uncertainty" or not parent or parent not in audit.index:
            continue
        if audit.loc[parent, "decision"] == "exclude" and row["decision"] == "include":
            audit.loc[column, "decision"] = "exclude"
            audit.loc[column, "exclusion_group"] = "parent_excluded"
            audit.loc[column, "reason"] = (
                f"Excluded because its parent measurement {parent} is excluded. "
                f"Parent reason: {audit.loc[parent, 'reason']}"
            )

    for column, row in audit[audit["decision"] == "include"].iterrows():
        if row["missing_fraction_all_rows"] == 1.0 or row["n_unique_non_missing"] <= 1:
            audit.loc[column, "decision"] = "exclude"
            audit.loc[column, "exclusion_group"] = "unavailable"
            audit.loc[column, "reason"] = (
                "No usable variation remains in this cumulative table."
            )
            continue
        if _structural_delivery_gap(frame, column, primary):
            audit.loc[column, "decision"] = "exclude"
            audit.loc[column, "exclusion_group"] = "structural_missingness"
            audit.loc[column, "reason"] = (
                row["reason"]
                + " Excluded from the proposed matrix because, among rows that have "
                "a transit depth, the column is entirely missing for at least one "
                "large koi_parm_prov delivery and present for another. Median "
                "imputation would assign one delivery's typical value to a different "
                "delivery. The column is not excluded because it encodes the disposition."
            ).strip()
            continue
        residual = float(row["missing_fraction_primary_rows"])
        if residual > MAX_RESIDUAL_MISSING_FRACTION:
            audit.loc[column, "decision"] = "exclude"
            audit.loc[column, "exclusion_group"] = "high_residual_missingness"
            audit.loc[column, "reason"] = (
                f"{row['reason']} Missing on {residual:.1%} of rows that have a "
                f"transit depth, above the {MAX_RESIDUAL_MISSING_FRACTION:.0%} "
                "residual-missing limit, so those gaps are not median-imputed."
            ).strip()
            continue
        if residual > 0:
            audit.loc[column, "impute_residual"] = True
            audit.loc[column, "reason"] = (
                f"{row['reason']} Missing on {residual:.1%} of rows that have a "
                "transit depth. Those residual gaps, and not the unfitted block, "
                "are median-imputed inside each development training fold."
            ).strip()

    pairs = _error_pairs(audit)
    for negative, positive in pairs:
        fraction = _symmetry_fraction(numeric[positive], numeric[negative], primary)
        if fraction is None:
            continue
        audit.loc[negative, "symmetric_negative_fraction"] = fraction
        audit.loc[positive, "symmetric_negative_fraction"] = fraction
        if (
            audit.loc[negative, "decision"] == "include"
            and audit.loc[positive, "decision"] == "include"
            and fraction >= SYMMETRIC_ERROR_FRACTION
        ):
            audit.loc[negative, "decision"] = "exclude"
            audit.loc[negative, "exclusion_group"] = "redundant_uncertainty"
            audit.loc[negative, "impute_residual"] = False
            audit.loc[negative, "reason"] = (
                f"On rows with both sides observed, {fraction:.1%} satisfy "
                f"{positive} = -{negative}. The negative uncertainty repeats the "
                f"positive uncertainty {positive}, which is retained."
            )

    by_name = audit.set_index("column")
    for column, row in audit.iterrows():
        parent = row["parent"]
        if parent and parent in by_name.index and bool(by_name.loc[parent, "provisional"]):
            audit.loc[column, "provisional"] = True

    audit["team_status"] = "proposed_pending_team_agreement"
    audit = audit.reset_index(drop=True)
    return audit


def _error_pairs(audit: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = []
    indexed = audit.set_index("column")
    for column in indexed.index:
        if column.endswith("_err2"):
            positive = column[:-1] + "1"
            if positive in indexed.index:
                pairs.append((column, positive))
    return pairs


def _leakage_reason(column: str, label: str, description: str) -> str:
    if column == "kepler_name":
        return (
            "The column page says a Kepler name identifies an object confirmed "
            "or validated as a planet, a step up from the planet-candidate "
            "designation. The name, and an indicator that the name is present, "
            "are excluded from the scientific feature set."
        )
    if column == "koi_score":
        return (
            "Disposition Score is defined as the fraction of Robovetter Monte Carlo "
            "iterations that yield a disposition of CANDIDATE. It is an output of "
            "the existing vetting decision."
        )
    if column == "koi_pdisposition":
        return (
            "Disposition Using Kepler Data is the pipeline flag for the most "
            "probable physical explanation, with values such as FALSE POSITIVE "
            "and CANDIDATE. It is a disposition field, not a raw measurement."
        )
    if column == "koi_comment":
        return (
            "The column page describes this text as the reason an object was given "
            "a false-positive disposition, including Robovetter flag shorthand. "
            "It directly states the vetting conclusion."
        )
    if column.startswith("koi_fpflag"):
        return (
            f"{label or column} is a project disposition flag. "
            "koi_fpflag_co is specifically the centroid-offset decision inferred "
            "from the pixel measurements; the measurements themselves are reviewed "
            "separately. These flags are excluded from the scientific feature set."
        )
    return description or f"{column} encodes the disposition or vetting decision."


def _metadata_reason(column: str, label: str, n_unique: int) -> str:
    if column in {"kepid", "kepoi_name", "koi_tce_plnt_num"}:
        return (
            f"{label or column} is an identifier. kepid is used only to keep every "
            "KOI of a host star inside one partition."
        )
    if column in {"koi_datalink_dvr", "koi_datalink_dvs"}:
        return "Relative path to a data-validation report, not a measurement."
    if column == "koi_quarters":
        return (
            "Bit string of which quarters were searched. It is observational "
            "coverage metadata, not a numeric transit measurement, and is not "
            "expanded into new features in this stage."
        )
    if column == "koi_fittype":
        return (
            "Fit-type label. The value none means the archive did not provide a "
            "planetary fit. It is used to investigate the missing-data block and "
            "is not a model feature."
        )
    if n_unique <= 1:
        return f"{label or column} has a single value in this file ({n_unique} distinct non-missing value)."
    return (
        f"{label or column} records provenance, vetting-process status, a model "
        "name, or a display string rather than the measurement used for classification."
    )


def _count_decision(info: dict) -> tuple[str, str, str]:
    matches_rows = info.get("fraction_equal_to_koi_rows", 0.0)
    matches_planets = info.get("fraction_equal_to_confirmed_or_candidate", 0.0)
    if matches_planets >= 0.99 and matches_planets > matches_rows:
        return (
            "exclude",
            "disposition_leakage",
            "The column page calls koi_count the number of planet candidates in "
            f"a system. On {matches_planets:.1%} of host stars it equals the number "
            "of CONFIRMED or CANDIDATE rows, so it encodes the target.",
        )
    if matches_rows >= 0.99:
        return (
            "include",
            "included",
            "The column page calls koi_count the number of planet candidates, but "
            f"on {matches_rows:.1%} of host stars the stored value equals the number "
            "of KOI rows, including false positives. It is used as system "
            "multiplicity, not as a count of the target class. The value is shared "
            "by every KOI of a star, so the grouped split matters.",
        )
    return (
        "exclude",
        "unclear_count_definition",
        "The column page calls koi_count the number of planet candidates, but the "
        f"values match KOI-row multiplicity on {matches_rows:.1%} of stars and "
        f"match CONFIRMED+CANDIDATE counts on {matches_planets:.1%} of stars. "
        "It is left out until that conflict is resolved.",
    )


def included_columns(audit: pd.DataFrame) -> list[str]:
    included = audit.loc[audit["decision"] == "include", "column"]
    return included.tolist()


def experimental_stat_sig_columns(audit: pd.DataFrame) -> list[str]:
    """Primary features plus the held-out flux-weighted significance statistic.

    The extra column is for a later comparison. It is not part of the primary
    leakage-aware set.
    """
    primary = included_columns(audit)
    if "koi_fwm_stat_sig" in primary:
        raise RuntimeError("koi_fwm_stat_sig is still in the primary feature set.")
    return primary + ["koi_fwm_stat_sig"]


def diagnostic_columns(audit: pd.DataFrame) -> list[str]:
    """Scientific columns plus the disposition fields used only as a leakage check."""
    extras = [
        "koi_pdisposition",
        "koi_score",
        "koi_fpflag_nt",
        "koi_fpflag_ss",
        "koi_fpflag_co",
        "koi_fpflag_ec",
        "has_kepler_name",
    ]
    return included_columns(audit) + extras
