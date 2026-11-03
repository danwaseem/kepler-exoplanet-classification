"""Development-only sensitivity check for provisional centroid measurements.

The selected models and the 76-feature primary set are not replaced.
The test partition is not scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score

from exoplanet.config import (
    CLASS_ORDER,
    DATA_PROCESSED,
    DOCS,
    GROUP_COLUMN,
    ID_COLUMN,
    RAW_CSV,
    ROOT,
    TARGET_COLUMN,
)
from exoplanet.stage2 import _load_inputs, development_only, make_tree_pipeline
from exoplanet.stage3 import STAGE3_RESULTS, fold_agreement
from exoplanet.stage4 import STAGE4_RESULTS, _classification_scores, make_forest_pipeline
from exoplanet.stage5 import centroid_columns

STAGE5B_RESULTS = ROOT / "results" / "person3" / "centroid_sensitivity"
STAGE5B_FIGURES = ROOT / "figures" / "person3" / "centroid_sensitivity"
STAT_SIG = "koi_fwm_stat_sig"
LEAKAGE = {
    "koi_pdisposition",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "kepler_name",
    "koi_comment",
    "kepid",
    TARGET_COLUMN,
}


def build_configurations(primary: list[str], centroids: list[str]) -> dict[str, dict]:
    """A is the saved primary set. B drops centroid offsets. C is diagnostic."""
    if STAT_SIG in primary or STAT_SIG in centroids:
        raise RuntimeError(f"{STAT_SIG} is not part of the primary centroid-offset set.")
    missing = [column for column in centroids if column not in primary]
    if missing:
        raise RuntimeError(f"Centroid columns are missing from the primary set: {missing}")
    if len(primary) != len(set(primary)) or len(centroids) != len(set(centroids)):
        raise RuntimeError("A feature list contains duplicates.")
    blocked = LEAKAGE & set(primary)
    if blocked:
        raise RuntimeError(f"The primary set includes leakage columns: {sorted(blocked)}")
    centroid_set = set(centroids)
    without = [column for column in primary if column not in centroid_set]
    return {
        "A": {
            "columns": list(primary),
            "role": "primary scientific set",
            "diagnostic": False,
        },
        "B": {
            "columns": without,
            "role": "primary set without provisional centroid offsets",
            "diagnostic": False,
        },
        "C": {
            "columns": list(primary) + [STAT_SIG],
            "role": "diagnostic addition of a potentially vetting-derived feature",
            "diagnostic": True,
        },
    }


def _recall_by_class(y_true, y_pred) -> dict[str, float]:
    values = recall_score(
        y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0
    )
    return {
        "recall_confirmed": float(values[0]),
        "recall_candidate": float(values[1]),
        "recall_false_positive": float(values[2]),
    }


def _evaluate(frame: pd.DataFrame, columns: list[str], factory) -> dict:
    frame = development_only(frame)
    blocked = (LEAKAGE | {GROUP_COLUMN}) & set(columns)
    if blocked:
        raise RuntimeError(f"A configuration includes blocked columns: {sorted(blocked)}")
    if frame.groupby(GROUP_COLUMN)["dev_fold"].nunique().max() != 1:
        raise RuntimeError("A host star appears in more than one development fold.")
    folds = frame["dev_fold"].to_numpy()
    if set(np.unique(folds)) != set(range(5)):
        raise RuntimeError(f"Expected development folds 0-4, found {sorted(set(folds))}.")
    x = frame[columns]
    y = frame[TARGET_COLUMN]
    oof = pd.Series(index=frame.index, dtype=object)
    fold_rows = []
    for fold_id in range(5):
        train = np.flatnonzero(folds != fold_id)
        valid = np.flatnonzero(folds == fold_id)
        train_groups = set(frame.iloc[train][GROUP_COLUMN])
        valid_groups = set(frame.iloc[valid][GROUP_COLUMN])
        if train_groups & valid_groups:
            raise RuntimeError(f"Fold {fold_id} shares a host star with its training rows.")
        pipeline = factory()
        pipeline.fit(x.iloc[train], y.iloc[train])
        pred_train = pipeline.predict(x.iloc[train])
        pred_valid = pipeline.predict(x.iloc[valid])
        oof.iloc[valid] = pred_valid
        train_scores = _classification_scores(y.iloc[train], pred_train)
        valid_scores = _classification_scores(y.iloc[valid], pred_valid)
        valid_recall = _recall_by_class(y.iloc[valid], pred_valid)
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": int(len(train)),
                "n_validation": int(len(valid)),
                "train_accuracy": train_scores["accuracy"],
                "train_macro_f1": train_scores["macro_f1"],
                "validation_accuracy": valid_scores["accuracy"],
                "validation_macro_f1": valid_scores["macro_f1"],
                **{f"validation_{name}": value for name, value in valid_recall.items()},
            }
        )
    if oof.isna().any():
        raise RuntimeError("An out-of-fold prediction is missing.")
    matrix = confusion_matrix(y, oof, labels=CLASS_ORDER)
    return {"folds": fold_rows, "confusion": matrix, "oof_recall": _recall_by_class(y, oof)}


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def _candidate_row(matrix: np.ndarray) -> dict[str, int]:
    row = matrix[CLASS_ORDER.index("CANDIDATE")]
    return {
        "support": int(row.sum()),
        "predicted_confirmed": int(row[0]),
        "predicted_candidate": int(row[1]),
        "predicted_false_positive": int(row[2]),
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _pm(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _attach_stat_sig(development: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Add the diagnostic column for development KOIs only."""
    raw = pd.read_csv(RAW_CSV, usecols=[ID_COLUMN, STAT_SIG])
    ids = set(development[ID_COLUMN].astype(str))
    kept = raw.loc[raw[ID_COLUMN].astype(str).isin(ids)].copy()
    kept[ID_COLUMN] = kept[ID_COLUMN].astype(str)
    if kept[ID_COLUMN].duplicated().any():
        raise RuntimeError(f"{STAT_SIG} is duplicated for a development KOI.")
    if set(kept[ID_COLUMN]) != ids:
        raise RuntimeError(f"{STAT_SIG} is missing for a development KOI.")
    frame = development.copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)
    merged = frame.merge(kept, on=ID_COLUMN, how="left", validate="one_to_one")
    if len(merged) != len(development) or merged["partition"].ne("development").any():
        raise RuntimeError("The diagnostic column changed the development row set.")
    development_only(merged)
    missing = float(merged[STAT_SIG].isna().mean())
    return merged, missing


def _load_selected() -> tuple[dict, dict]:
    tree = json.loads((STAGE3_RESULTS / "model_config.json").read_text(encoding="utf-8"))
    forest = json.loads((STAGE4_RESULTS / "model_config.json").read_text(encoding="utf-8"))
    return tree["selected_hyperparameters"], forest["selected_hyperparameters"]


def _factories(tree_params: dict, forest_params: dict) -> dict:
    def tree_factory():
        return make_tree_pipeline(
            tree_params["max_depth"],
            min_samples_split=int(tree_params["min_samples_split"]),
            min_samples_leaf=int(tree_params["min_samples_leaf"]),
            ccp_alpha=float(tree_params["ccp_alpha"]),
        )

    def forest_factory():
        return make_forest_pipeline(
            {
                "n_estimators": int(forest_params["n_estimators"]),
                "max_depth": forest_params["max_depth"],
                "max_features": forest_params["max_features"],
                "min_samples_leaf": int(forest_params["min_samples_leaf"]),
                "class_weight": forest_params["class_weight"],
            }
        )

    return {"Pruned tree": tree_factory, "Tuned forest": forest_factory}


def _summary_row(model: str, config: str, info: dict, result: dict) -> dict:
    folds = result["folds"]
    row = {
        "model": model,
        "configuration": config,
        "role": info["role"],
        "diagnostic": info["diagnostic"],
        "n_features": len(info["columns"]),
    }
    for metric in (
        "validation_accuracy",
        "validation_macro_f1",
        "validation_recall_confirmed",
        "validation_recall_candidate",
        "validation_recall_false_positive",
        "train_macro_f1",
    ):
        mean, std = _mean_std([fold[metric] for fold in folds])
        row[f"mean_{metric}"] = mean
        row[f"std_{metric}"] = std
    for name, value in result["oof_recall"].items():
        row[f"oof_{name}"] = value
    counts = _candidate_row(result["confusion"])
    row.update({f"oof_candidate_{name}": value for name, value in counts.items()})
    return row


def _paired(model: str, contrast: str, base: list[dict], other: list[dict], metric: str) -> dict:
    deltas = [other[index][metric] - base[index][metric] for index in range(5)]
    agreement = fold_agreement(deltas)
    return {
        "model": model,
        "contrast": contrast,
        "metric": metric,
        **{key: value for key, value in agreement.items() if key != "deltas"},
        "fold_0": deltas[0],
        "fold_1": deltas[1],
        "fold_2": deltas[2],
        "fold_3": deltas[3],
        "fold_4": deltas[4],
    }


def _fold_count(count: int) -> str:
    noun = "fold" if int(count) == 1 else "folds"
    return f"{int(count)} {noun}"


def _agreement_sentence(row: dict) -> str:
    return (
        f"Mean paired difference {_fmt(row['mean_delta'])} "
        f"(std {_fmt(row['std_delta'])}), from {_fmt(row['min_delta'])} to {_fmt(row['max_delta'])}. "
        f"The alternative is higher on {_fold_count(row['n_higher'])}, "
        f"lower on {_fold_count(row['n_lower'])}, "
        f"and equal on {_fold_count(row['n_equal'])}. This count is not a hypothesis test."
    )


def _plot_grouped(summary: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    configs = ["A", "B", "C"]
    models = ["Pruned tree", "Tuned forest"]
    colors = {"Pruned tree": "#009E73", "Tuned forest": "#0072B2"}
    x = np.arange(len(configs))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    for offset, model in zip((-width / 2, width / 2), models):
        part = summary.loc[summary["model"].eq(model)].set_index("configuration").loc[configs]
        axis.bar(
            x + offset,
            part[f"mean_{metric}"],
            width,
            yerr=part[f"std_{metric}"],
            color=colors[model],
            capsize=3,
            label=model,
        )
    axis.set_xticks(x, ["A primary", "B no centroid offsets", "C diagnostic"])
    axis.set_ylabel(ylabel)
    axis.set_title("Development grouped validation")
    axis.legend(frameon=False)
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_deltas(paired: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), sharey=True)
    colors = {"B-A": "#0072B2", "C-A": "#E69F00"}
    for axis, model in zip(axes, ["Pruned tree", "Tuned forest"]):
        for contrast in ("B-A", "C-A"):
            row = paired.loc[
                paired["model"].eq(model)
                & paired["contrast"].eq(contrast)
                & paired["metric"].eq("validation_macro_f1")
            ].iloc[0]
            values = [row[f"fold_{fold}"] for fold in range(5)]
            axis.plot(range(5), values, marker="o", color=colors[contrast], label=contrast)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(5), [f"Fold {fold}" for fold in range(5)], rotation=20)
        axis.set_title(model)
        axis.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Validation macro-F1 minus configuration A")
    axes[1].legend(frameon=False, title="C is diagnostic")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion(results: dict, model: str, configs: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(configs), figsize=(4.6 * len(configs), 4.6))
    for axis, config in zip(axes, configs):
        matrix = results[(model, config)]["confusion"]
        label = {"A": "primary", "B": "no centroid offsets", "C": "diagnostic"}[config]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_ORDER,
            yticklabels=CLASS_ORDER,
            ax=axis,
            cbar=False,
        )
        axis.set_xlabel("Predicted disposition")
        axis.set_ylabel("Archive disposition")
        axis.set_title(f"{model}, {config} ({label})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _cell(result: dict, true_label: str, predicted: str) -> int:
    matrix = result["confusion"]
    return int(matrix[CLASS_ORDER.index(true_label), CLASS_ORDER.index(predicted)])


def _block_sentence(results: dict) -> str:
    def span(model: str, true_label: str, predicted: str) -> str:
        before = _cell(results[(model, "A")], true_label, predicted)
        after = _cell(results[(model, "B")], true_label, predicted)
        if before == after:
            return f"stay at {before}"
        return f"go from {before} to {after}"

    return (
        "Removing the centroid block does not increase the number of Candidate rows predicted as Confirmed. "
        f"Those pooled counts {span('Pruned tree', 'CANDIDATE', 'CONFIRMED')} for the pruned tree "
        f"and {span('Tuned forest', 'CANDIDATE', 'CONFIRMED')} for the tuned forest. "
        "The Candidate recall loss appears as False Positive predictions: they "
        f"{span('Pruned tree', 'CANDIDATE', 'FALSE POSITIVE')} for the tree and "
        f"{span('Tuned forest', 'CANDIDATE', 'FALSE POSITIVE')} for the forest. "
        "The change is broader than the Candidate row. For the tuned forest, False Positive "
        f"rows predicted Confirmed {span('Tuned forest', 'FALSE POSITIVE', 'CONFIRMED')}, "
        "False Positive rows predicted Candidate "
        f"{span('Tuned forest', 'FALSE POSITIVE', 'CANDIDATE')}, and Confirmed rows "
        f"predicted False Positive {span('Tuned forest', 'CONFIRMED', 'FALSE POSITIVE')}."
    )


def _confusion_sentence(model: str, left: dict, right: dict, right_name: str) -> str:
    left_counts = _candidate_row(left["confusion"])
    right_counts = _candidate_row(right["confusion"])
    recall_left = left["oof_recall"]["recall_candidate"]
    recall_right = right["oof_recall"]["recall_candidate"]
    return (
        f"For the {model}, pooled development out-of-fold Candidate recall is "
        f"{_fmt(recall_left)} under A and {_fmt(recall_right)} under {right_name}. "
        f"Candidate rows predicted Confirmed change from {left_counts['predicted_confirmed']} "
        f"to {right_counts['predicted_confirmed']}. Candidate rows predicted False Positive "
        f"change from {left_counts['predicted_false_positive']} to "
        f"{right_counts['predicted_false_positive']}."
    )


def _write_findings(payload: dict, path: Path) -> None:
    summary_lines = [
        "| Model | Config | Features | Validation accuracy | Validation macro-F1 | Confirmed recall | Candidate recall | False Positive recall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        label = f"{row['configuration']}{' (diagnostic)' if row['diagnostic'] else ''}"
        summary_lines.append(
            f"| {row['model']} | {label} | {int(row['n_features'])} | "
            f"{_pm(row['mean_validation_accuracy'], row['std_validation_accuracy'])} | "
            f"{_pm(row['mean_validation_macro_f1'], row['std_validation_macro_f1'])} | "
            f"{_pm(row['mean_validation_recall_confirmed'], row['std_validation_recall_confirmed'])} | "
            f"{_pm(row['mean_validation_recall_candidate'], row['std_validation_recall_candidate'])} | "
            f"{_pm(row['mean_validation_recall_false_positive'], row['std_validation_recall_false_positive'])} |"
        )
    paired_lines = [
        "| Model | Contrast | Metric | Mean difference | Std | Min | Max | Higher | Lower |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["paired"]:
        paired_lines.append(
            f"| {row['model']} | {row['contrast']} | {row['metric']} | {_fmt(row['mean_delta'])} | "
            f"{_fmt(row['std_delta'])} | {_fmt(row['min_delta'])} | {_fmt(row['max_delta'])} | "
            f"{int(row['n_higher'])} | {int(row['n_lower'])} |"
        )
    text = "\n".join(
        [
            "# Stage 5B: centroid-offset sensitivity",
            "",
            "This is a limited sensitivity check on the development partition. "
            "It does not replace the 76-feature primary set or the selected tree and forest. "
            "Configuration C adds `koi_fwm_stat_sig` as a diagnostic feature. "
            "That column is a potentially vetting-derived summary, not an accepted production feature. "
            "The test partition is not scored.",
            "",
            "## Configurations",
            "",
            f"- A: the existing primary set, {payload['n_features_a']} columns.",
            f"- B: A without {payload['n_centroid_measurements']} provisional centroid-offset "
            f"measurements and their {payload['n_centroid_uncertainties']} uncertainty columns, "
            f"leaving {payload['n_features_b']} columns.",
            "- C: A plus `koi_fwm_stat_sig`. Diagnostic only. The column page says this percent "
            "indicates whether there is a statistically significant flux-weighted offset between "
            "in-transit and out-of-transit images, and that 100% means there is no offset and "
            "there is confidence that the transit is on the target star.",
            f"- Development rows: {payload['n_rows']:,}. Host stars: {payload['n_groups']:,}.",
            f"- Saved split SHA-256: `{payload['split_sha256']}`",
            f"- Missing fraction of `koi_fwm_stat_sig` on development rows: {_fmt(payload['stat_sig_missing'])}. "
            "The median imputer is fit inside each training fold.",
            "- Hyperparameters are the Stage 3 pruned tree and the Stage 4 tuned forest. "
            "Configuration A reproduces their saved grouped validation macro-F1.",
            "",
            "## Grouped validation",
            "",
            "Means and standard deviations are across the five saved development folds.",
            "",
            *summary_lines,
            "",
            "## Paired differences relative to A",
            "",
            "Each difference is the named configuration minus A on the same validation fold.",
            "",
            *paired_lines,
            "",
            *payload["paired_sentences"],
            "",
            "## Candidate rows",
            "",
            "The question is whether removing the centroid offsets changes Candidate recall "
            "or moves Candidate rows into Confirmed or False Positive. Counts are pooled "
            "out-of-fold predictions on the same 7,381 development rows.",
            "",
            *payload["confusion_sentences"],
            "",
            payload["candidate_block_sentence"],
            "",
            "Configuration C remains a diagnostic experiment. A higher or lower score with "
            "`koi_fwm_stat_sig` included does not make that column part of the primary feature set. "
            "On these folds its mean validation macro-F1 is slightly below configuration A for both models, "
            "and the fold differences change sign.",
            "",
            "## Limitations",
            "",
            "The same development folds were used to select the hyperparameters, so these "
            "scores are not a fresh confirmation of that selection. Fold counts are not a "
            "hypothesis test. Correlated centroid columns can share a signal, so removing the "
            "whole block is a different question from permuting one column. The primary models "
            "and the primary feature configuration were not changed.",
            "",
            "## Not done in this stage",
            "",
            "The test partition was not scored. No new primary model or feature list was selected.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _check_configuration_a(results: dict) -> None:
    saved_tree = pd.read_csv(STAGE3_RESULTS / "comparison_folds.csv")
    saved_tree = saved_tree.loc[saved_tree["model"].eq("selected")].sort_values("fold")
    tree_scores = [row["validation_macro_f1"] for row in results[("Pruned tree", "A")]["folds"]]
    if not np.allclose(tree_scores, saved_tree["validation_macro_f1"], atol=1e-8):
        raise RuntimeError("Configuration A does not reproduce the pruned-tree validation macro-F1.")
    saved_forest = pd.read_csv(STAGE4_RESULTS / "comparison_folds.csv")
    saved_forest = saved_forest.loc[saved_forest["model"].eq("Tuned forest")].sort_values("fold")
    forest_scores = [row["validation_macro_f1"] for row in results[("Tuned forest", "A")]["folds"]]
    if not np.allclose(forest_scores, saved_forest["validation_macro_f1"], atol=1e-8):
        raise RuntimeError("Configuration A does not reproduce the tuned-forest validation macro-F1.")
    saved_matrix = pd.read_csv(STAGE4_RESULTS / "oof_confusion_matrix.csv", index_col=0)
    saved_matrix = saved_matrix.loc[CLASS_ORDER, CLASS_ORDER].to_numpy()
    if not np.array_equal(results[("Tuned forest", "A")]["confusion"], saved_matrix):
        raise RuntimeError("Configuration A does not reproduce the tuned-forest confusion matrix.")
    per_class = pd.read_csv(STAGE4_RESULTS / "per_class_metrics.csv")
    for model in ("Pruned tree", "Tuned forest"):
        for label, key in (
            ("CONFIRMED", "recall_confirmed"),
            ("CANDIDATE", "recall_candidate"),
            ("FALSE POSITIVE", "recall_false_positive"),
        ):
            saved = float(
                per_class.loc[per_class["model"].eq(model) & per_class["class"].eq(label), "recall"].iloc[0]
            )
            observed = results[(model, "A")]["oof_recall"][key]
            if abs(saved - observed) > 1e-8:
                raise RuntimeError(f"Configuration A {model} {label} recall does not match Stage 4.")


def main() -> None:
    STAGE5B_RESULTS.mkdir(parents=True, exist_ok=True)
    STAGE5B_FIGURES.mkdir(parents=True, exist_ok=True)
    development, primary, digest = _load_inputs()
    development = development_only(development)
    if len(development) != 7381 or len(primary) != 76:
        raise RuntimeError(
            f"Expected 7381 development rows and 76 features, found {len(development)} and {len(primary)}."
        )
    feature_table = pd.read_csv(DATA_PROCESSED / "proposed_leakage_aware_features.csv")
    centroids = centroid_columns(feature_table)
    configs = build_configurations(primary, centroids)
    frame, missing_stat_sig = _attach_stat_sig(development)
    tree_params, forest_params = _load_selected()
    factories = _factories(tree_params, forest_params)
    results = {}
    for model, factory in factories.items():
        for name, info in configs.items():
            print(f"{model} configuration {name}", flush=True)
            results[(model, name)] = _evaluate(frame, info["columns"], factory)
            score = np.mean([row["validation_macro_f1"] for row in results[(model, name)]["folds"]])
            print(f"  mean validation macro-F1 {score:.4f}", flush=True)
    _check_configuration_a(results)

    summary = pd.DataFrame(
        [
            _summary_row(model, name, configs[name], results[(model, name)])
            for model in factories
            for name in configs
        ]
    )
    paired_rows = []
    for model in factories:
        base = results[(model, "A")]["folds"]
        for name in ("B", "C"):
            other = results[(model, name)]["folds"]
            for metric in (
                "validation_macro_f1",
                "validation_accuracy",
                "validation_recall_candidate",
                "validation_recall_confirmed",
                "validation_recall_false_positive",
            ):
                paired_rows.append(_paired(model, f"{name}-A", base, other, metric))
    paired = pd.DataFrame(paired_rows)
    paired_sentences = []
    for _, row in paired.loc[paired["metric"].eq("validation_macro_f1")].iterrows():
        paired_sentences.append(f"- {row['model']}, {row['contrast']} macro-F1: {_agreement_sentence(row)}")
    for _, row in paired.loc[paired["metric"].eq("validation_recall_candidate")].iterrows():
        paired_sentences.append(
            f"- {row['model']}, {row['contrast']} Candidate recall: {_agreement_sentence(row)}"
        )
    confusion_sentences = []
    for model in factories:
        confusion_sentences.append(
            _confusion_sentence(model, results[(model, "A")], results[(model, "B")], "B")
        )
        confusion_sentences.append(
            _confusion_sentence(model, results[(model, "A")], results[(model, "C")], "diagnostic C")
        )
    payload = {
        "n_rows": int(len(development)),
        "n_groups": int(development[GROUP_COLUMN].nunique()),
        "n_features_a": len(configs["A"]["columns"]),
        "n_features_b": len(configs["B"]["columns"]),
        "n_centroids": len(centroids),
        "n_centroid_measurements": int((feature_table["column"].isin(centroids) & feature_table["feature_family"].eq("vetting_measurement")).sum()),
        "n_centroid_uncertainties": int((feature_table["column"].isin(centroids) & feature_table["feature_family"].eq("uncertainty")).sum()),
        "candidate_block_sentence": _block_sentence(results),
        "split_sha256": digest,
        "stat_sig_missing": missing_stat_sig,
        "summary": summary.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
        "paired_sentences": paired_sentences,
        "confusion_sentences": confusion_sentences,
    }
    summary.to_csv(STAGE5B_RESULTS / "summary.csv", index=False)
    paired.to_csv(STAGE5B_RESULTS / "paired_differences.csv", index=False)
    fold_rows = []
    confusion_rows = []
    for model in factories:
        for name in configs:
            for fold in results[(model, name)]["folds"]:
                fold_rows.append({"model": model, "configuration": name, "diagnostic": configs[name]["diagnostic"], **fold})
            matrix = results[(model, name)]["confusion"]
            for true_label, row in zip(CLASS_ORDER, matrix):
                for predicted, count in zip(CLASS_ORDER, row):
                    confusion_rows.append(
                        {
                            "model": model,
                            "configuration": name,
                            "diagnostic": configs[name]["diagnostic"],
                            "true_class": true_label,
                            "predicted_class": predicted,
                            "count": int(count),
                        }
                    )
    pd.DataFrame(fold_rows).to_csv(STAGE5B_RESULTS / "fold_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(STAGE5B_RESULTS / "oof_confusion.csv", index=False)
    config = {
        "test_scored": False,
        "models_changed": False,
        "primary_feature_configuration_changed": False,
        "split_sha256": digest,
        "tree_hyperparameters": tree_params,
        "forest_hyperparameters": forest_params,
        "configurations": {
            name: {
                "role": info["role"],
                "diagnostic": info["diagnostic"],
                "n_features": len(info["columns"]),
                "columns": info["columns"],
            }
            for name, info in configs.items()
        },
        "centroid_columns_removed_in_B": centroids,
        "configuration_C_note": (
            "Diagnostic only. koi_fwm_stat_sig is a potentially vetting-derived feature "
            "and is not an accepted production feature."
        ),
    }
    (STAGE5B_RESULTS / "model_config.json").write_text(
        json.dumps(_json_ready(config), indent=2), encoding="utf-8"
    )
    _plot_grouped(
        summary,
        "validation_macro_f1",
        "Mean grouped validation macro-F1",
        STAGE5B_FIGURES / "validation_macro_f1.png",
    )
    _plot_grouped(
        summary,
        "validation_recall_candidate",
        "Mean grouped validation Candidate recall",
        STAGE5B_FIGURES / "candidate_recall.png",
    )
    _plot_deltas(paired, STAGE5B_FIGURES / "paired_macro_f1_differences.png")
    _plot_confusion(
        results,
        "Tuned forest",
        ["A", "B"],
        STAGE5B_FIGURES / "forest_confusion_A_vs_B.png",
    )
    _plot_confusion(
        results,
        "Pruned tree",
        ["A", "B"],
        STAGE5B_FIGURES / "tree_confusion_A_vs_B.png",
    )
    _plot_confusion(
        results,
        "Tuned forest",
        ["A", "C"],
        STAGE5B_FIGURES / "forest_confusion_A_vs_C_diagnostic.png",
    )
    _write_findings(payload, DOCS / "stage5b_findings.md")
    print("Stage 5B complete. Primary models and the primary feature set were not changed.")


if __name__ == "__main__":
    main()
