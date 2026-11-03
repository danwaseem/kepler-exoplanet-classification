"""Decision Tree tuning and cost-complexity pruning on the development partition.

The saved grouped split is reused. The test partition is not scored.
The search maximizes mean grouped validation macro-F1 on a pre-specified
grid. A forest is not trained.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from exoplanet.config import (
    DATA_PROCESSED,
    DOCS,
    GROUP_COLUMN,
    RANDOM_SEED,
    ROOT,
    TARGET_COLUMN,
)
from exoplanet.stage2 import (
    STAGE2_RESULTS,
    _load_inputs,
    _mean_std,
    _scores,
    assert_primary_features,
    development_only,
    make_tree_pipeline,
)

STAGE3_RESULTS = ROOT / "results" / "person3" / "decision_tree" / "stage3"
STAGE3_FIGURES = ROOT / "figures" / "person3" / "decision_tree" / "stage3"

STRUCTURAL_MAX_DEPTHS = [6, 7, 8, 9, 10, 12]
STRUCTURAL_MIN_SAMPLES_SPLIT = [2, 10, 40]
STRUCTURAL_MIN_SAMPLES_LEAF = [1, 5, 20]
TOP_STRUCTURAL = 3
CCP_PERCENTILES = 11
CCP_GEOM_POINTS = 12

UNRESTRICTED = {
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "ccp_alpha": 0.0,
}
DEPTH8 = {
    "max_depth": 8,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "ccp_alpha": 0.0,
}


def same_params(left: dict, right: dict) -> bool:
    return (
        left["max_depth"] == right["max_depth"]
        and int(left["min_samples_split"]) == int(right["min_samples_split"])
        and int(left["min_samples_leaf"]) == int(right["min_samples_leaf"])
        and abs(float(left["ccp_alpha"]) - float(right["ccp_alpha"]))
        <= 1e-12 * max(1.0, abs(float(left["ccp_alpha"])))
    )


def params_label(params: dict) -> str:
    depth = "unrestricted" if params["max_depth"] is None else str(int(params["max_depth"]))
    return (
        f"max_depth={depth}, min_samples_split={int(params['min_samples_split'])}, "
        f"min_samples_leaf={int(params['min_samples_leaf'])}, "
        f"ccp_alpha={float(params['ccp_alpha']):.6g}"
    )


def select_configuration(rows: list[dict]) -> dict:
    """Pick the maximum mean grouped validation macro-F1.

    Exact ties break toward the smaller training-validation macro-F1 gap,
    then toward fewer leaves. A higher mean is the selection rule, not a
    significance test.
    """
    if not rows:
        raise RuntimeError("No configurations were evaluated.")
    return max(
        rows,
        key=lambda row: (
            float(row["mean_validation_macro_f1"]),
            -float(row["mean_macro_f1_gap"]),
            -float(row["mean_n_leaves"]),
        ),
    )


def fold_agreement(deltas: list[float], tolerance: float = 1e-12) -> dict:
    """Count how many folds move in each direction. This is not a hypothesis test."""
    if not deltas:
        raise RuntimeError("Fold differences are empty.")
    higher = sum(delta > tolerance for delta in deltas)
    lower = sum(delta < -tolerance for delta in deltas)
    values = np.asarray(deltas, dtype=float)
    return {
        "n_folds": len(deltas),
        "n_higher": int(higher),
        "n_lower": int(lower),
        "n_equal": int(len(deltas) - higher - lower),
        "mean_delta": float(values.mean()),
        "std_delta": float(values.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "min_delta": float(values.min()),
        "max_delta": float(values.max()),
        "deltas": [float(delta) for delta in deltas],
    }


def _increasing_positive(values: list[float], rtol: float = 1e-5) -> list[float]:
    chosen: list[float] = []
    for value in sorted(float(item) for item in values if np.isfinite(item) and item > 0):
        if not chosen or abs(value - chosen[-1]) > rtol * max(chosen[-1], value, 1e-15):
            chosen.append(value)
    return chosen


def build_shared_alphas(paths: list[np.ndarray]) -> list[float]:
    """One alpha grid from training-fold pruning paths.

    The values are medians, across folds, of fixed percentiles of each fold's
    positive training-path alphas, plus a log spacing between the median
    minimum and median maximum. Validation scores are not used.
    """
    positive_paths = [np.asarray(path, dtype=float) for path in paths]
    positive_paths = [path[path > 0] for path in positive_paths]
    positive_paths = [path for path in positive_paths if len(path)]
    if not positive_paths:
        raise RuntimeError("The training-fold pruning paths have no positive alpha.")
    percentiles = np.linspace(0, 100, CCP_PERCENTILES)
    anchors: list[float] = []
    for percentile in percentiles:
        per_fold = [float(np.percentile(path, percentile)) for path in positive_paths]
        anchors.append(float(np.median(per_fold)))
    lo = float(np.median([path.min() for path in positive_paths]))
    hi = float(np.median([path.max() for path in positive_paths]))
    if hi > lo:
        anchors.extend(float(value) for value in np.geomspace(lo, hi, CCP_GEOM_POINTS))
    else:
        anchors.append(lo)
    return [0.0] + _increasing_positive(anchors)


def evaluate_configuration(frame: pd.DataFrame, feature_columns: list[str], params: dict) -> dict:
    """Fit one configuration independently on each saved development fold."""
    frame = development_only(frame)
    assert_primary_features(feature_columns)
    if float(params["ccp_alpha"]) < 0:
        raise RuntimeError("ccp_alpha must be non-negative.")
    folds = frame["dev_fold"].to_numpy()
    x = frame[feature_columns]
    y = frame[TARGET_COLUMN]
    fold_rows = []
    for fold_id in range(5):
        train = folds != fold_id
        valid = folds == fold_id
        if frame.loc[train, GROUP_COLUMN].isin(frame.loc[valid, GROUP_COLUMN]).any():
            raise RuntimeError(f"Fold {fold_id} shares a host star with its training rows.")
        pipeline = make_tree_pipeline(
            params["max_depth"],
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            ccp_alpha=float(params["ccp_alpha"]),
        )
        pipeline.fit(x.loc[train], y.loc[train])
        tree = pipeline.named_steps["tree"]
        train_scores = _scores(y.loc[train], pipeline.predict(x.loc[train]))
        valid_scores = _scores(y.loc[valid], pipeline.predict(x.loc[valid]))
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": int(train.sum()),
                "n_validation": int(valid.sum()),
                "train_accuracy": train_scores["accuracy"],
                "train_macro_f1": train_scores["macro_f1"],
                "validation_accuracy": valid_scores["accuracy"],
                "validation_macro_f1": valid_scores["macro_f1"],
                "depth": int(tree.get_depth()),
                "n_leaves": int(tree.get_n_leaves()),
            }
        )
    summary = {
        "max_depth": params["max_depth"],
        "min_samples_split": int(params["min_samples_split"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "ccp_alpha": float(params["ccp_alpha"]),
        "mean_macro_f1_gap": float(
            np.mean([row["train_macro_f1"] for row in fold_rows])
            - np.mean([row["validation_macro_f1"] for row in fold_rows])
        ),
        "mean_depth": float(np.mean([row["depth"] for row in fold_rows])),
        "mean_n_leaves": float(np.mean([row["n_leaves"] for row in fold_rows])),
        "folds": fold_rows,
    }
    for metric in ("train_accuracy", "validation_accuracy", "train_macro_f1", "validation_macro_f1"):
        stats = _mean_std([row[metric] for row in fold_rows])
        summary[f"mean_{metric}"] = stats["mean"]
        summary[f"std_{metric}"] = stats["std"]
    summary["mean_depth_std"] = _mean_std([row["depth"] for row in fold_rows])["std"]
    summary["mean_n_leaves_std"] = _mean_std([row["n_leaves"] for row in fold_rows])["std"]
    return summary


def training_ccp_alphas(frame: pd.DataFrame, feature_columns: list[str], params: dict) -> list[np.ndarray]:
    """Pruning-path alphas from each training fold. Validation rows are not used."""
    frame = development_only(frame)
    folds = frame["dev_fold"].to_numpy()
    x = frame[feature_columns]
    y = frame[TARGET_COLUMN]
    paths = []
    for fold_id in range(5):
        train = folds != fold_id
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x_train = imputer.fit_transform(x.loc[train])
        tree = make_tree_pipeline(
            params["max_depth"],
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            ccp_alpha=0.0,
        ).named_steps["tree"]
        path = tree.cost_complexity_pruning_path(x_train, y.loc[train])
        paths.append(np.asarray(path.ccp_alphas, dtype=float))
    return paths


def _cache_key(params: dict) -> tuple:
    depth = params["max_depth"]
    return (
        None if depth is None else int(depth),
        int(params["min_samples_split"]),
        int(params["min_samples_leaf"]),
        float(params["ccp_alpha"]),
    )


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _fmt_delta(value: float) -> str:
    text = f"{value:.4f}"
    if text in {"0.0000", "-0.0000"} and abs(value) > 1e-12:
        return f"{value:.6f}"
    return text


def _agreement_sentence(baseline: str, agreement: dict) -> str:
    return (
        f"Against the {baseline} tree, the selected configuration has higher validation "
        f"macro-F1 on {agreement['n_higher']} of {agreement['n_folds']} folds, "
        f"lower on {agreement['n_lower']}, and equal on {agreement['n_equal']}. "
        f"The paired differences, selected minus {baseline}, have mean "
        f"{_fmt_delta(agreement['mean_delta'])}, minimum {_fmt_delta(agreement['min_delta'])}, "
        f"and maximum {_fmt_delta(agreement['max_delta'])}. "
        "No hypothesis test was applied. A higher mean is not reported as statistically significant."
    )


def _ccp_statement(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda row: row["ccp_alpha"])
    base = next(row for row in ordered if row["ccp_alpha"] == 0.0)
    best = max(ordered, key=lambda row: row["mean_validation_macro_f1"])
    last = ordered[-1]
    sentences = [
        (
            f"At ccp_alpha=0 the unrestricted tree has mean training macro-F1 "
            f"{_fmt(base['mean_train_macro_f1'])} and mean validation macro-F1 "
            f"{_fmt(base['mean_validation_macro_f1'])}."
        ),
        (
            f"On this shared alpha grid, the highest mean validation macro-F1 is "
            f"{_fmt(best['mean_validation_macro_f1'])} at ccp_alpha={best['ccp_alpha']:.6g}, "
            f"where mean training macro-F1 is {_fmt(best['mean_train_macro_f1'])} "
            f"and the mean leaf count is {best['mean_n_leaves']:.1f}."
        ),
        (
            f"At the largest evaluated alpha ({last['ccp_alpha']:.6g}), mean training macro-F1 is "
            f"{_fmt(last['mean_train_macro_f1'])} and mean validation macro-F1 is "
            f"{_fmt(last['mean_validation_macro_f1'])}."
        ),
    ]
    train_drop = base["mean_train_macro_f1"] - last["mean_train_macro_f1"]
    if train_drop > 0:
        sentences.append(
            f"Raising ccp_alpha from 0 to {last['ccp_alpha']:.6g} lowers mean training macro-F1 by "
            f"{_fmt(train_drop)}."
        )
    val_change = best["mean_validation_macro_f1"] - base["mean_validation_macro_f1"]
    if best["ccp_alpha"] == 0.0:
        sentences.append(
            "The highest grouped validation macro-F1 on this pruning grid is the unpruned unrestricted tree."
        )
    else:
        sentences.append(
            f"ccp_alpha={best['ccp_alpha']:.6g} changes grouped validation macro-F1 by "
            f"{_fmt_delta(val_change)} relative to the unpruned unrestricted tree."
        )
    return " ".join(sentences)


def _plot_ccp(rows: list[dict], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: row["ccp_alpha"])
    alpha = np.asarray([row["ccp_alpha"] for row in ordered], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    specs = (
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
    )
    for axis, (metric, label) in zip(axes, specs):
        axis.errorbar(
            alpha,
            [row[f"mean_train_{metric}"] for row in ordered],
            yerr=[row[f"std_train_{metric}"] for row in ordered],
            marker="o",
            color="#0072B2",
            label="Training folds",
        )
        axis.errorbar(
            alpha,
            [row[f"mean_validation_{metric}"] for row in ordered],
            yerr=[row[f"std_validation_{metric}"] for row in ordered],
            marker="o",
            color="#D55E00",
            label="Grouped validation folds",
        )
        axis.set_xlabel("ccp_alpha")
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.set_ylim(0, 1.05)
        axis.grid(True, alpha=0.3)
    positive = alpha[alpha > 0]
    if len(positive):
        for axis in axes:
            axis.set_xscale("symlog", linthresh=float(positive.min()))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Cost-complexity pruning of the unrestricted tree")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_comparison(models: list[dict], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(models))
    width = 0.36
    train = [model["mean_train_macro_f1"] for model in models]
    valid = [model["mean_validation_macro_f1"] for model in models]
    train_err = [model["std_train_macro_f1"] for model in models]
    valid_err = [model["std_validation_macro_f1"] for model in models]
    axis.bar(x - width / 2, train, width, yerr=train_err, color="#0072B2", label="Training macro-F1", capsize=3)
    axis.bar(
        x + width / 2,
        valid,
        width,
        yerr=valid_err,
        color="#D55E00",
        label="Grouped validation macro-F1",
        capsize=3,
    )
    axis.set_xticks(x, [model["role"] for model in models])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Macro-F1")
    axis.set_title("Development grouped cross-validation")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _metric_cells(row: dict) -> str:
    return (
        f"{_fmt(row['mean_train_accuracy'])} ± {_fmt(row['std_train_accuracy'])} | "
        f"{_fmt(row['mean_validation_accuracy'])} ± {_fmt(row['std_validation_accuracy'])} | "
        f"{_fmt(row['mean_train_macro_f1'])} ± {_fmt(row['std_train_macro_f1'])} | "
        f"{_fmt(row['mean_validation_macro_f1'])} ± {_fmt(row['std_validation_macro_f1'])} | "
        f"{_fmt(row['mean_macro_f1_gap'])} | {row['mean_depth']:.2f} | {row['mean_n_leaves']:.1f}"
    )


def _write_findings(payload: dict, path: Path) -> None:
    header = (
        "| Model | Train accuracy | Validation accuracy | Train macro-F1 | Validation macro-F1 | "
        "Macro-F1 gap | Mean depth | Mean leaves |"
    )
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    comparison_lines = [header, sep]
    for model in payload["comparison"]:
        comparison_lines.append(f"| {model['role']} | {_metric_cells(model)} |")
    top_lines = [header, sep]
    for model in payload["top_configurations"]:
        top_lines.append(f"| {params_label(model)} | {_metric_cells(model)} |")
    fold_header = (
        "| Fold | Unrestricted | Depth 8 | Selected | Selected − unrestricted | Selected − depth 8 |"
    )
    fold_sep = "| ---: | ---: | ---: | ---: | ---: | ---: |"
    fold_lines = [fold_header, fold_sep]
    for row in payload["fold_macro_f1"]:
        fold_lines.append(
            f"| {row['fold']} | {_fmt(row['unrestricted'])} | {_fmt(row['depth_8'])} | "
            f"{_fmt(row['selected'])} | {_fmt_delta(row['selected_minus_unrestricted'])} | "
            f"{_fmt_delta(row['selected_minus_depth_8'])} |"
        )
    ccp_lines = [
        "| ccp_alpha | Train accuracy | Validation accuracy | Train macro-F1 | Validation macro-F1 | Mean leaves |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["ccp_curve"]:
        ccp_lines.append(
            f"| {row['ccp_alpha']:.6g} | {_fmt(row['mean_train_accuracy'])} ± {_fmt(row['std_train_accuracy'])} "
            f"| {_fmt(row['mean_validation_accuracy'])} ± {_fmt(row['std_validation_accuracy'])} "
            f"| {_fmt(row['mean_train_macro_f1'])} ± {_fmt(row['std_train_macro_f1'])} "
            f"| {_fmt(row['mean_validation_macro_f1'])} ± {_fmt(row['std_validation_macro_f1'])} "
            f"| {row['mean_n_leaves']:.1f} |"
        )
    selected = payload["selected"]
    text = "\n".join(
        [
            "# Stage 3: Decision Tree tuning and cost-complexity pruning",
            "",
            "The feature list is still provisional. This stage uses the same 76 primary features, "
            "the 7,381 development rows with a measured transit depth, and the saved `dev_fold` "
            "assignments. The imputer is fit inside each training fold. The test partition is not "
            "scored. The experimental `koi_fwm_stat_sig` configuration is not trained. "
            "A forest is not trained. No configuration is declared statistically significant.",
            "",
            "## Sample and search",
            "",
            f"- Development rows: {payload['n_rows']:,}",
            f"- Development host stars: {payload['n_groups']:,}",
            f"- Primary features: {payload['n_features']}",
            f"- Saved split SHA-256: `{payload['split_sha256']}`",
            f"- Seed: {payload['random_seed']}",
            f"- Structural grid: max_depth {STRUCTURAL_MAX_DEPTHS}, "
            f"min_samples_split {STRUCTURAL_MIN_SAMPLES_SPLIT}, "
            f"min_samples_leaf {STRUCTURAL_MIN_SAMPLES_LEAF}, ccp_alpha 0 "
            f"({payload['n_structural']} configurations).",
            f"- Cost-complexity grid: {payload['n_alphas']} shared alphas, applied to the "
            "unrestricted tree, the depth-8 tree, and the three structural configurations "
            "with the highest mean validation macro-F1. Overlapping configurations are "
            "fit once. Alpha candidates are summaries of the training-fold pruning paths, "
            "not of the validation scores.",
            f"- Unique configurations evaluated: {payload['n_configurations']}.",
            "",
            "The selection rule is the maximum mean grouped validation macro-F1. "
            "If two means are equal, the smaller training-minus-validation macro-F1 gap wins, "
            "then the smaller mean leaf count. The reported validation score of the selected "
            "configuration is the score used to choose it. It is not a nested estimate on "
            "observations held out from the search. Depth 8 was the highest point on the "
            "Stage 2 curve for these same folds, so a gain over that tree is not a confirmation "
            "on new data.",
            "",
            "## Selected configuration",
            "",
            params_label(selected),
            "",
            f"Mean depth {selected['mean_depth']:.2f} (std {selected['mean_depth_std']:.2f}). "
            f"Mean leaves {selected['mean_n_leaves']:.1f} (std {selected['mean_n_leaves_std']:.1f}). "
            f"Macro-F1 gap, training minus grouped validation: {_fmt(selected['mean_macro_f1_gap'])}.",
            "",
            payload["selected_identity"],
            "",
            "A tree with these hyperparameters was also fit on all development rows so the "
            "configuration can be inspected. That fit is in-sample. It was not scored on the test partition.",
            "",
            f"- In-sample training accuracy: {_fmt(payload['development_fit']['train_accuracy'])}",
            f"- In-sample training macro-F1: {_fmt(payload['development_fit']['train_macro_f1'])}",
            f"- In-sample depth: {payload['development_fit']['depth']}",
            f"- In-sample leaves: {payload['development_fit']['n_leaves']}",
            "",
            "## Comparison",
            "",
            "The depth-8 tree uses `min_samples_split=2`, `min_samples_leaf=1`, and `ccp_alpha=0`, "
            "matching the Stage 2 curve. The unrestricted tree uses the same defaults with no depth limit.",
            "",
            *comparison_lines,
            "",
            "### Fold-level validation macro-F1",
            "",
            *fold_lines,
            "",
            payload["agreement_unrestricted"],
            "",
            payload["agreement_depth8"],
            "",
            "## Other evaluated configurations",
            "",
            "The five highest mean grouped validation macro-F1 values. The full grid is "
            "`results/person3/decision_tree/stage3/search_summary.csv`.",
            "",
            *top_lines,
            "",
            "## Cost-complexity pruning",
            "",
            "The curve below is the unrestricted tree at each shared alpha. "
            "Error bars in the figure are the sample standard deviation across the five folds.",
            "",
            *ccp_lines,
            "",
            payload["ccp_statement"],
            "",
            "## Tradeoff",
            "",
            payload["tradeoff"],
            "",
            _neighbor_note(payload["top_configurations"]),
            "",
            "## Not done in this stage",
            "",
            "The test partition was not scored. The `koi_fwm_stat_sig` configuration was not trained. "
            "The 363 incomplete KOIs were not fit. A forest was not trained. "
            "The search did not cross every structural combination with every alpha. "
            "No significance test was used to choose the configuration.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _neighbor_note(top: list[dict]) -> str:
    if len(top) < 2:
        return ""
    selected = top[0]
    neighbor = top[1]
    margin = float(selected["mean_validation_macro_f1"]) - float(neighbor["mean_validation_macro_f1"])
    sentences = [
        (
            f"The next configuration, {params_label(neighbor)}, has mean validation macro-F1 "
            f"{_fmt(neighbor['mean_validation_macro_f1'])}, mean validation accuracy "
            f"{_fmt(neighbor['mean_validation_accuracy'])}, a macro-F1 gap of "
            f"{_fmt(neighbor['mean_macro_f1_gap'])}, and {neighbor['mean_n_leaves']:.1f} mean leaves. "
            f"The selected configuration leads that neighbor by {_fmt_delta(margin)} macro-F1."
        )
    ]
    grouped: dict[tuple, list[int]] = {}
    for row in top:
        key = (
            row["max_depth"],
            int(row["min_samples_leaf"]),
            round(float(row["ccp_alpha"]), 12),
            round(float(row["mean_validation_macro_f1"]), 10),
        )
        grouped.setdefault(key, []).append(int(row["min_samples_split"]))
    for (depth, leaf, _alpha, score), splits in grouped.items():
        unique_splits = sorted(set(splits))
        if len(unique_splits) < 2:
            continue
        depth_label = "unrestricted" if depth is None else str(depth)
        sentences.append(
            f"At max_depth={depth_label} and min_samples_leaf={leaf}, min_samples_split values "
            f"{unique_splits} produce the same grouped validation macro-F1 ({_fmt(score)}). "
            "A minimum leaf size of that order already limits the splits those settings can make."
        )
    return " ".join(sentences)


def _identity_sentence(selected: dict, unrestricted: dict, depth8: dict) -> str:
    if same_params(selected, unrestricted):
        return "The search selected the unrestricted tree. No evaluated configuration had a higher mean grouped validation macro-F1."
    if same_params(selected, depth8):
        return "The search selected the depth-8 tree. No evaluated configuration had a higher mean grouped validation macro-F1."
    return (
        "The search selected a different configuration from both the unrestricted tree and the depth-8 tree."
    )


def _tradeoff_sentence(selected: dict, unrestricted: dict, depth8: dict) -> str:
    sentences = [
        (
            f"The selected tree's mean validation macro-F1 is {_fmt(selected['mean_validation_macro_f1'])}, "
            f"compared with {_fmt(depth8['mean_validation_macro_f1'])} for the depth-8 tree and "
            f"{_fmt(unrestricted['mean_validation_macro_f1'])} for the unrestricted tree."
        ),
        (
            f"Its macro-F1 gap is {_fmt(selected['mean_macro_f1_gap'])}, compared with "
            f"{_fmt(depth8['mean_macro_f1_gap'])} at depth 8 and {_fmt(unrestricted['mean_macro_f1_gap'])} "
            "when the tree is unrestricted."
        ),
        (
            f"Mean leaves are {selected['mean_n_leaves']:.1f} for the selected tree, "
            f"{depth8['mean_n_leaves']:.1f} at depth 8, and {unrestricted['mean_n_leaves']:.1f} "
            "for the unrestricted tree."
        ),
    ]
    val_vs_depth = selected["mean_validation_macro_f1"] - depth8["mean_validation_macro_f1"]
    gap_vs_depth = selected["mean_macro_f1_gap"] - depth8["mean_macro_f1_gap"]
    if same_params(selected, depth8):
        sentences.append(
            "Tuning did not find a higher grouped validation macro-F1 than the depth-8 tree "
            "with the default leaf settings and no cost-complexity penalty."
        )
    elif val_vs_depth > 0 and gap_vs_depth < 0:
        sentences.append(
            "Relative to depth 8, the selected tree has a higher grouped validation macro-F1 "
            "and a smaller training-validation gap on these folds."
        )
    elif val_vs_depth > 0 and gap_vs_depth > 0:
        sentences.append(
            "Relative to depth 8, the selected tree has a higher grouped validation macro-F1 "
            "and a larger training-validation gap on these folds."
        )
    elif val_vs_depth == 0:
        sentences.append(
            "The selected tree matches the depth-8 grouped validation macro-F1."
        )
    return " ".join(sentences)


def _assert_stage2(unrestricted: dict, depth8: dict) -> None:
    saved = pd.read_csv(STAGE2_RESULTS / "fold_metrics.csv")
    for saved_row, fold in zip(saved.to_dict(orient="records"), unrestricted["folds"], strict=True):
        for metric in ("train_accuracy", "validation_accuracy", "train_macro_f1", "validation_macro_f1"):
            if not np.isclose(fold[metric], saved_row[metric], atol=1e-10):
                raise RuntimeError(
                    f"Unrestricted fold {fold['fold']} {metric} does not match Stage 2."
                )
    curve = pd.read_csv(STAGE2_RESULTS / "validation_curve.csv")
    depth_row = curve.loc[(curve["max_depth"] == 8) & (~curve["unrestricted"].astype(bool))].iloc[0]
    for metric in ("train_accuracy", "validation_accuracy", "train_macro_f1", "validation_macro_f1"):
        if not np.isclose(depth8[f"mean_{metric}"], depth_row[f"mean_{metric}"], atol=1e-10):
            raise RuntimeError(f"Depth-8 mean {metric} does not match the Stage 2 validation curve.")


def _summary_frame(rows: list[dict]) -> pd.DataFrame:
    public = []
    for row in rows:
        public.append(
            {
                "max_depth": "None" if row["max_depth"] is None else int(row["max_depth"]),
                "min_samples_split": row["min_samples_split"],
                "min_samples_leaf": row["min_samples_leaf"],
                "ccp_alpha": row["ccp_alpha"],
                "mean_train_accuracy": row["mean_train_accuracy"],
                "std_train_accuracy": row["std_train_accuracy"],
                "mean_validation_accuracy": row["mean_validation_accuracy"],
                "std_validation_accuracy": row["std_validation_accuracy"],
                "mean_train_macro_f1": row["mean_train_macro_f1"],
                "std_train_macro_f1": row["std_train_macro_f1"],
                "mean_validation_macro_f1": row["mean_validation_macro_f1"],
                "std_validation_macro_f1": row["std_validation_macro_f1"],
                "mean_macro_f1_gap": row["mean_macro_f1_gap"],
                "mean_depth": row["mean_depth"],
                "std_depth": row["mean_depth_std"],
                "mean_n_leaves": row["mean_n_leaves"],
                "std_n_leaves": row["mean_n_leaves_std"],
            }
        )
    return pd.DataFrame(public).sort_values(
        ["mean_validation_macro_f1", "mean_macro_f1_gap"],
        ascending=[False, True],
    )


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _base_params(row: dict) -> dict:
    return {
        "max_depth": row["max_depth"],
        "min_samples_split": int(row["min_samples_split"]),
        "min_samples_leaf": int(row["min_samples_leaf"]),
        "ccp_alpha": 0.0,
    }


def main() -> None:
    STAGE3_RESULTS.mkdir(parents=True, exist_ok=True)
    STAGE3_FIGURES.mkdir(parents=True, exist_ok=True)
    development, columns, digest = _load_inputs()
    development = development_only(development)
    if len(development) != 7381 or len(columns) != 76:
        raise RuntimeError(
            f"Expected 7381 development rows and 76 features, found {len(development)} and {len(columns)}."
        )
    if development.groupby(GROUP_COLUMN)["dev_fold"].nunique().max() != 1:
        raise RuntimeError("A host star appears in more than one development fold.")
    if set(development["dev_fold"].unique()) != set(range(5)):
        raise RuntimeError("The saved development folds are not 0 through 4.")

    cache: dict[tuple, dict] = {}

    def get_config(params: dict) -> dict:
        key = _cache_key(params)
        if key not in cache:
            print(f"Evaluating {params_label(params)}", flush=True)
            cache[key] = evaluate_configuration(development, columns, params)
        return cache[key]

    unrestricted = get_config(UNRESTRICTED)
    structural_rows = []
    for depth, split, leaf in product(
        STRUCTURAL_MAX_DEPTHS,
        STRUCTURAL_MIN_SAMPLES_SPLIT,
        STRUCTURAL_MIN_SAMPLES_LEAF,
    ):
        structural_rows.append(
            get_config(
                {
                    "max_depth": depth,
                    "min_samples_split": split,
                    "min_samples_leaf": leaf,
                    "ccp_alpha": 0.0,
                }
            )
        )
    depth8 = get_config(DEPTH8)
    _assert_stage2(unrestricted, depth8)

    ranked = sorted(structural_rows, key=lambda row: row["mean_validation_macro_f1"], reverse=True)
    top_structural = []
    for row in ranked:
        base = _base_params(row)
        if any(same_params(base, existing) for existing in top_structural):
            continue
        top_structural.append(base)
        if len(top_structural) >= TOP_STRUCTURAL:
            break
    families = []
    for base in [UNRESTRICTED, DEPTH8, *top_structural]:
        if any(same_params(base, existing) for existing in families):
            continue
        families.append(base)

    paths = training_ccp_alphas(development, columns, UNRESTRICTED)
    alphas = build_shared_alphas(paths)
    for family in families:
        for alpha in alphas:
            params = dict(family)
            params["ccp_alpha"] = float(alpha)
            get_config(params)

    evaluated = list(cache.values())
    selected = select_configuration(evaluated)
    unrestricted_ccp = [
        row
        for row in evaluated
        if row["max_depth"] is None
        and row["min_samples_split"] == 2
        and row["min_samples_leaf"] == 1
    ]
    agreement_unrestricted = fold_agreement(
        [
            selected_fold["validation_macro_f1"] - base_fold["validation_macro_f1"]
            for selected_fold, base_fold in zip(selected["folds"], unrestricted["folds"], strict=True)
        ]
    )
    agreement_depth8 = fold_agreement(
        [
            selected_fold["validation_macro_f1"] - base_fold["validation_macro_f1"]
            for selected_fold, base_fold in zip(selected["folds"], depth8["folds"], strict=True)
        ]
    )
    fold_table = []
    for selected_fold, unrestricted_fold, depth_fold in zip(
        selected["folds"], unrestricted["folds"], depth8["folds"], strict=True
    ):
        fold_table.append(
            {
                "fold": selected_fold["fold"],
                "unrestricted": unrestricted_fold["validation_macro_f1"],
                "depth_8": depth_fold["validation_macro_f1"],
                "selected": selected_fold["validation_macro_f1"],
                "selected_minus_unrestricted": (
                    selected_fold["validation_macro_f1"] - unrestricted_fold["validation_macro_f1"]
                ),
                "selected_minus_depth_8": (
                    selected_fold["validation_macro_f1"] - depth_fold["validation_macro_f1"]
                ),
            }
        )

    pipeline = make_tree_pipeline(
        selected["max_depth"],
        min_samples_split=selected["min_samples_split"],
        min_samples_leaf=selected["min_samples_leaf"],
        ccp_alpha=selected["ccp_alpha"],
    )
    pipeline.fit(development[columns], development[TARGET_COLUMN])
    in_sample_prediction = pipeline.predict(development[columns])
    in_sample_scores = _scores(development[TARGET_COLUMN], in_sample_prediction)
    development_fit = {
        "train_accuracy": in_sample_scores["accuracy"],
        "train_macro_f1": in_sample_scores["macro_f1"],
        "depth": int(pipeline.named_steps["tree"].get_depth()),
        "n_leaves": int(pipeline.named_steps["tree"].get_n_leaves()),
        "scored_on_test": False,
        "role": "in_sample_development_fit",
    }

    def role_row(role: str, row: dict) -> dict:
        return {"role": role, **row}

    comparison = [
        role_row("Unrestricted", unrestricted),
        role_row("Depth 8", depth8),
        role_row("Selected", selected),
    ]
    top = sorted(
        evaluated,
        key=lambda row: (row["mean_validation_macro_f1"], -row["mean_macro_f1_gap"]),
        reverse=True,
    )[:5]
    ccp_statement = _ccp_statement(unrestricted_ccp)
    payload = {
        "stage": 3,
        "team_finalized": False,
        "random_seed": RANDOM_SEED,
        "split_sha256": digest,
        "partition": "development",
        "test_scored": False,
        "n_rows": int(len(development)),
        "n_groups": int(development[GROUP_COLUMN].nunique()),
        "n_features": len(columns),
        "n_structural": len(structural_rows),
        "n_alphas": len(alphas),
        "n_configurations": len(evaluated),
        "alphas": alphas,
        "selected": selected,
        "selected_identity": _identity_sentence(selected, unrestricted, depth8),
        "comparison": comparison,
        "top_configurations": top,
        "fold_macro_f1": fold_table,
        "agreement_unrestricted": _agreement_sentence("unrestricted", agreement_unrestricted),
        "agreement_depth8": _agreement_sentence("depth-8", agreement_depth8),
        "agreement": {
            "versus_unrestricted": agreement_unrestricted,
            "versus_depth_8": agreement_depth8,
        },
        "ccp_curve": sorted(unrestricted_ccp, key=lambda row: row["ccp_alpha"]),
        "ccp_statement": ccp_statement,
        "tradeoff": _tradeoff_sentence(selected, unrestricted, depth8),
        "development_fit": development_fit,
        "selection_rule": (
            "maximum mean grouped validation macro-F1; "
            "exact ties break toward the smaller macro-F1 gap, then fewer leaves"
        ),
    }
    summary = _summary_frame(evaluated)
    summary.to_csv(STAGE3_RESULTS / "search_summary.csv", index=False)
    _summary_frame(payload["ccp_curve"]).sort_values("ccp_alpha").to_csv(
        STAGE3_RESULTS / "ccp_curve.csv", index=False
    )
    pd.DataFrame(fold_table).to_csv(STAGE3_RESULTS / "fold_consistency.csv", index=False)
    fold_records = []
    for role, row in (("unrestricted", unrestricted), ("depth_8", depth8), ("selected", selected)):
        for fold in row["folds"]:
            fold_records.append({"model": role, **fold})
    pd.DataFrame(fold_records).to_csv(STAGE3_RESULTS / "comparison_folds.csv", index=False)
    config = {
        "team_finalized": False,
        "feature_configuration": "primary_leakage_aware_provisional",
        "n_features": len(columns),
        "feature_columns": columns,
        "split_file": "data/processed/proposed_splits.csv",
        "split_sha256": digest,
        "test_scored": False,
        "random_seed": RANDOM_SEED,
        "structural_max_depth": STRUCTURAL_MAX_DEPTHS,
        "structural_min_samples_split": STRUCTURAL_MIN_SAMPLES_SPLIT,
        "structural_min_samples_leaf": STRUCTURAL_MIN_SAMPLES_LEAF,
        "ccp_alphas": alphas,
        "selection_rule": payload["selection_rule"],
        "selected_hyperparameters": {
            "criterion": "gini",
            "max_depth": selected["max_depth"],
            "min_samples_split": selected["min_samples_split"],
            "min_samples_leaf": selected["min_samples_leaf"],
            "ccp_alpha": selected["ccp_alpha"],
            "class_weight": None,
            "random_state": RANDOM_SEED,
            "imputer": "median",
        },
        "mean_validation_macro_f1": selected["mean_validation_macro_f1"],
        "mean_macro_f1_gap": selected["mean_macro_f1_gap"],
    }
    _json(STAGE3_RESULTS / "model_config.json", config)
    _json(STAGE3_RESULTS / "stage3_metrics.json", payload)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": columns,
            "hyperparameters": config["selected_hyperparameters"],
            "role": "in_sample_development_fit",
            "test_scored": False,
            "random_seed": RANDOM_SEED,
        },
        STAGE3_RESULTS / "selected_development_tree.joblib",
    )
    _plot_ccp(payload["ccp_curve"], STAGE3_FIGURES / "ccp_alpha_curve.png")
    _plot_comparison(comparison, STAGE3_FIGURES / "model_comparison.png")
    _write_findings(payload, DOCS / "stage3_findings.md")
    print(
        "Stage 3 complete: "
        f"selected {params_label(selected)}; "
        f"validation macro-F1 {selected['mean_validation_macro_f1']:.4f}; "
        f"gap {selected['mean_macro_f1_gap']:.4f}; "
        f"mean leaves {selected['mean_n_leaves']:.1f}."
    )
    print(payload["selected_identity"])
    print(payload["tradeoff"])
    print(payload["agreement_unrestricted"])
    print(payload["agreement_depth8"])


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
