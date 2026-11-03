"""Random forest baseline, staged tuning, and comparison on the development partition.

The saved grouped split is reused. The test partition is not scored.
The search maximizes mean grouped validation macro-F1. It does not evaluate
every combination of the forest hyperparameters.
"""

from __future__ import annotations

import json
import os
import time
from itertools import product
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from exoplanet.config import (
    CLASS_ORDER,
    DOCS,
    GROUP_COLUMN,
    RANDOM_SEED,
    ROOT,
    TARGET_COLUMN,
)
from exoplanet.stage2 import (
    _load_inputs,
    _mean_std,
    assert_primary_features,
    development_only,
    make_tree_pipeline,
)
from exoplanet.stage3 import STAGE3_RESULTS, fold_agreement

STAGE4_RESULTS = ROOT / "results" / "stage4"
STAGE4_FIGURES = ROOT / "figures" / "stage4"

N_ESTIMATORS_GRID = [100, 200, 400]
MAX_DEPTH_GRID = [None, 8, 12, 20]
MAX_FEATURES_GRID = ["sqrt", 0.5, 1.0]
MIN_SAMPLES_LEAF_GRID = [1, 2, 5]
CLASS_WEIGHT_GRID = [None, "balanced"]
N_RANDOM = 16
TOP_PROMOTE = 3

BASELINE = {
    "n_estimators": 100,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "class_weight": None,
}
UNRESTRICTED_TREE = {
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "ccp_alpha": 0.0,
}
DEPTH8_TREE = {
    "max_depth": 8,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "ccp_alpha": 0.0,
}


def forest_grid() -> list[dict]:
    """Every listed combination. The search evaluates only a staged subset."""
    rows = []
    for n_estimators, max_depth, max_features, min_samples_leaf, class_weight in product(
        N_ESTIMATORS_GRID,
        MAX_DEPTH_GRID,
        MAX_FEATURES_GRID,
        MIN_SAMPLES_LEAF_GRID,
        CLASS_WEIGHT_GRID,
    ):
        rows.append(
            {
                "n_estimators": int(n_estimators),
                "max_depth": max_depth,
                "max_features": max_features,
                "min_samples_leaf": int(min_samples_leaf),
                "class_weight": class_weight,
            }
        )
    return rows


def forest_key(params: dict) -> tuple:
    features = params["max_features"]
    if not isinstance(features, str):
        features = float(features)
    depth = params["max_depth"]
    return (
        int(params["n_estimators"]),
        None if depth is None else int(depth),
        features,
        int(params["min_samples_leaf"]),
        params["class_weight"],
    )


def random_phase(n_configs: int = N_RANDOM, seed: int = RANDOM_SEED) -> list[dict]:
    """Reproducible sample of the 100-tree grid, excluding the baseline cell.

    The sample is drawn before any validation score is computed.
    """
    pool = [params for params in forest_grid() if params["n_estimators"] == 100]
    pool = [params for params in pool if forest_key(params) != forest_key(BASELINE)]
    if n_configs >= len(pool):
        raise RuntimeError("The random phase is not smaller than the 100-tree grid.")
    generator = np.random.RandomState(seed)
    chosen = sorted(generator.choice(len(pool), size=n_configs, replace=False).tolist())
    return [pool[index] for index in chosen]


def select_forest(rows: list[dict]) -> dict:
    """Maximum mean grouped validation macro-F1.

    Exact ties break toward the smaller macro-F1 gap, then fewer trees.
    A higher mean is the selection rule, not a significance test.
    """
    if not rows:
        raise RuntimeError("No forest configurations were evaluated.")
    return max(
        rows,
        key=lambda row: (
            float(row["mean_validation_macro_f1"]),
            -float(row["mean_macro_f1_gap"]),
            -int(row["n_estimators"]),
        ),
    )


def make_forest_pipeline(params: dict) -> Pipeline:
    """Median imputation plus a forest. The imputer is fit with the pipeline."""
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=int(params["n_estimators"]),
                    criterion="gini",
                    max_depth=params["max_depth"],
                    min_samples_split=2,
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    max_features=params["max_features"],
                    bootstrap=True,
                    class_weight=params["class_weight"],
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _classification_scores(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    labels = CLASS_ORDER
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
    }


def cross_validate_pipeline(
    frame: pd.DataFrame,
    feature_columns: list[str],
    factory,
) -> dict:
    """Fit a new pipeline on each saved development fold and keep out-of-fold predictions."""
    frame = development_only(frame)
    assert_primary_features(feature_columns)
    folds = frame["dev_fold"].to_numpy()
    x = frame[feature_columns]
    y = frame[TARGET_COLUMN]
    oof = pd.Series(index=frame.index, dtype=object)
    fold_rows = []
    for fold_id in range(5):
        train = folds != fold_id
        valid = folds == fold_id
        if frame.loc[train, GROUP_COLUMN].isin(frame.loc[valid, GROUP_COLUMN]).any():
            raise RuntimeError(f"Fold {fold_id} shares a host star with its training rows.")
        pipeline = factory()
        started = time.perf_counter()
        pipeline.fit(x.loc[train], y.loc[train])
        fit_seconds = time.perf_counter() - started
        pred_train = pipeline.predict(x.loc[train])
        pred_valid = pipeline.predict(x.loc[valid])
        oof.loc[frame.index[valid]] = pred_valid
        train_scores = _classification_scores(y.loc[train], pred_train)
        valid_scores = _classification_scores(y.loc[valid], pred_valid)
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": int(train.sum()),
                "n_validation": int(valid.sum()),
                "fit_seconds": float(fit_seconds),
                **{f"train_{name}": value for name, value in train_scores.items()},
                **{f"validation_{name}": value for name, value in valid_scores.items()},
            }
        )
    if oof.isna().any():
        raise RuntimeError("An out-of-fold prediction is missing.")
    summary = {
        "folds": fold_rows,
        "oof_prediction": oof,
        "total_fit_seconds": float(sum(row["fit_seconds"] for row in fold_rows)),
        "mean_fit_seconds": float(np.mean([row["fit_seconds"] for row in fold_rows])),
        "mean_macro_f1_gap": float(
            np.mean([row["train_macro_f1"] for row in fold_rows])
            - np.mean([row["validation_macro_f1"] for row in fold_rows])
        ),
    }
    for metric in (
        "train_accuracy",
        "validation_accuracy",
        "train_macro_precision",
        "validation_macro_precision",
        "train_macro_recall",
        "validation_macro_recall",
        "train_macro_f1",
        "validation_macro_f1",
        "fit_seconds",
    ):
        stats = _mean_std([row[metric] for row in fold_rows])
        summary[f"mean_{metric}"] = stats["mean"]
        summary[f"std_{metric}"] = stats["std"]
    return summary


def _per_class(y_true: pd.Series, y_pred: pd.Series) -> list[dict]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        zero_division=0,
    )
    rows = []
    for index, label in enumerate(CLASS_ORDER):
        rows.append(
            {
                "class": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )
    return rows


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _fmt_delta(value: float) -> str:
    text = f"{value:.4f}"
    if text in {"0.0000", "-0.0000"} and abs(value) > 1e-12:
        return f"{value:.6f}"
    return text


def forest_label(params: dict) -> str:
    depth = "None" if params["max_depth"] is None else str(int(params["max_depth"]))
    weight = "None" if params["class_weight"] is None else str(params["class_weight"])
    features = params["max_features"]
    feature_text = features if isinstance(features, str) else f"{float(features):.1f}"
    return (
        f"n_estimators={int(params['n_estimators'])}, max_depth={depth}, "
        f"max_features={feature_text}, min_samples_leaf={int(params['min_samples_leaf'])}, "
        f"class_weight={weight}"
    )


def _metric_row(row: dict) -> str:
    return (
        f"{_fmt(row['mean_train_accuracy'])} ± {_fmt(row['std_train_accuracy'])} | "
        f"{_fmt(row['mean_validation_accuracy'])} ± {_fmt(row['std_validation_accuracy'])} | "
        f"{_fmt(row['mean_validation_macro_precision'])} ± {_fmt(row['std_validation_macro_precision'])} | "
        f"{_fmt(row['mean_validation_macro_recall'])} ± {_fmt(row['std_validation_macro_recall'])} | "
        f"{_fmt(row['mean_train_macro_f1'])} ± {_fmt(row['std_train_macro_f1'])} | "
        f"{_fmt(row['mean_validation_macro_f1'])} ± {_fmt(row['std_validation_macro_f1'])} | "
        f"{_fmt(row['mean_macro_f1_gap'])} | {row['total_fit_seconds']:.1f}"
    )


def _gap_statement(tuned: dict, pruned: dict, depth8: dict, unrestricted: dict, baseline: dict) -> str:
    sentences = [
        (
            f"Macro-F1 gaps, training minus grouped validation, are "
            f"{_fmt(unrestricted['mean_macro_f1_gap'])} for the unrestricted tree, "
            f"{_fmt(depth8['mean_macro_f1_gap'])} for the depth-8 tree, "
            f"{_fmt(pruned['mean_macro_f1_gap'])} for the pruned tree, "
            f"{_fmt(baseline['mean_macro_f1_gap'])} for the baseline forest, and "
            f"{_fmt(tuned['mean_macro_f1_gap'])} for the tuned forest."
        )
    ]
    gap_smaller = tuned["mean_macro_f1_gap"] < pruned["mean_macro_f1_gap"]
    val_higher = tuned["mean_validation_macro_f1"] > pruned["mean_validation_macro_f1"]
    if gap_smaller and val_higher:
        sentences.append(
            "On these development folds, the tuned forest has a higher grouped validation "
            "macro-F1 and a smaller training-validation macro-F1 gap than the pruned tree. "
            "That is the measured sense in which this forest shows less overfitting than that tree."
        )
    elif gap_smaller:
        sentences.append(
            "The tuned forest's training-validation macro-F1 gap is smaller than the pruned "
            "tree's gap, and its grouped validation macro-F1 is not higher. The smaller gap "
            "does not by itself show an improvement over the pruned tree."
        )
    elif val_higher:
        sentences.append(
            "The tuned forest's grouped validation macro-F1 is higher than the pruned tree's, "
            "and its training-validation macro-F1 gap is not smaller."
        )
    else:
        sentences.append(
            "The tuned forest's grouped validation macro-F1 is not higher than the pruned "
            "tree's, and its training-validation macro-F1 gap is not smaller."
        )
    sentences.append("No hypothesis test was applied.")
    return " ".join(sentences)


def _selection_tradeoff(comparison: list[dict], per_class: list[dict], top_forests: list[dict]) -> str:
    tuned = next(model for model in comparison if model["role"] == "Tuned forest")
    baseline = next(model for model in comparison if model["role"] == "Baseline forest")
    sentences = [
        (
            f"The tuned forest's mean validation accuracy is {_fmt(tuned['mean_validation_accuracy'])}. "
            f"The baseline forest's mean validation accuracy is {_fmt(baseline['mean_validation_accuracy'])}."
        )
    ]
    if tuned["mean_validation_accuracy"] < baseline["mean_validation_accuracy"] and (
        tuned["mean_validation_macro_f1"] > baseline["mean_validation_macro_f1"]
    ):
        sentences.append(
            "Accuracy is lower for the tuned forest while macro-F1 is higher, which is why "
            "the macro-F1 rule selects it."
        )
    same_family = [
        row
        for row in top_forests
        if row["max_depth"] == tuned["max_depth"]
        and row["max_features"] == tuned["max_features"]
        and int(row["min_samples_leaf"]) == int(tuned["min_samples_leaf"])
        and row["class_weight"] == tuned["class_weight"]
    ]
    if len(same_family) >= 2:
        ordered = sorted(same_family, key=lambda row: int(row["n_estimators"]))
        detail = ", ".join(
            f"{int(row['n_estimators'])} trees at {_fmt(row['mean_validation_macro_f1'])}"
            for row in ordered
        )
        sentences.append(
            f"The same hyperparameters at different forest sizes scored {detail}. "
            "A larger forest did not raise validation macro-F1 for this setting."
            if ordered[-1]["mean_validation_macro_f1"] <= ordered[0]["mean_validation_macro_f1"]
            else f"The same hyperparameters at different forest sizes scored {detail}."
        )
    def class_f1(model: str, label: str, field: str) -> float:
        return next(
            row[field]
            for row in per_class
            if row["model"] == model and row["class"] == label
        )
    sentences.append(
        "Pooled out-of-fold candidate recall is "
        f"{_fmt(class_f1('Tuned forest', 'CANDIDATE', 'recall'))} for the tuned forest and "
        f"{_fmt(class_f1('Baseline forest', 'CANDIDATE', 'recall'))} for the baseline forest. "
        "Candidate precision is "
        f"{_fmt(class_f1('Tuned forest', 'CANDIDATE', 'precision'))} for the tuned forest and "
        f"{_fmt(class_f1('Baseline forest', 'CANDIDATE', 'precision'))} for the baseline forest."
    )
    return " ".join(sentences)


def _agreement_sentence(baseline_name: str, agreement: dict) -> str:
    return (
        f"Against the {baseline_name}, the tuned forest has higher validation macro-F1 on "
        f"{agreement['n_higher']} of {agreement['n_folds']} folds, lower on {agreement['n_lower']}, "
        f"and equal on {agreement['n_equal']}. The paired differences have mean "
        f"{_fmt_delta(agreement['mean_delta'])}, minimum {_fmt_delta(agreement['min_delta'])}, "
        f"and maximum {_fmt_delta(agreement['max_delta'])}. "
        "No hypothesis test was applied. A higher mean is not reported as statistically significant."
    )


def _plot_comparison(models: list[dict], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(models))
    width = 0.36
    axis.bar(
        x - width / 2,
        [model["mean_train_macro_f1"] for model in models],
        width,
        yerr=[model["std_train_macro_f1"] for model in models],
        color="#0072B2",
        label="Training macro-F1",
        capsize=3,
    )
    axis.bar(
        x + width / 2,
        [model["mean_validation_macro_f1"] for model in models],
        width,
        yerr=[model["std_validation_macro_f1"] for model in models],
        color="#D55E00",
        label="Grouped validation macro-F1",
        capsize=3,
    )
    axis.set_xticks(x, [model["role"] for model in models], rotation=15, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Macro-F1")
    axis.set_title("Development grouped cross-validation")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion(matrix: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
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
    axis.set_title("Tuned forest, development out-of-fold predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_findings(payload: dict, path: Path) -> None:
    header = (
        "| Model | Training accuracy | Validation accuracy | Validation macro precision | "
        "Validation macro recall | Training macro-F1 | Validation macro-F1 | Macro-F1 gap | Fit seconds |"
    )
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    comparison = [header, sep]
    for model in payload["comparison"]:
        comparison.append(f"| {model['role']} | {_metric_row(model)} |")
    tuning = [
        "| Configuration | Validation macro-F1 | Training macro-F1 | Macro-F1 gap | Fit seconds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in payload["top_forests"]:
        tuning.append(
            f"| {forest_label(model)} | {_fmt(model['mean_validation_macro_f1'])} "
            f"± {_fmt(model['std_validation_macro_f1'])} | {_fmt(model['mean_train_macro_f1'])} "
            f"| {_fmt(model['mean_macro_f1_gap'])} | {model['total_fit_seconds']:.1f} |"
        )
    fold_header = (
        "| Fold | Unrestricted tree | Depth-8 tree | Pruned tree | Baseline forest | Tuned forest |"
    )
    fold_lines = [fold_header, "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in payload["fold_macro_f1"]:
        fold_lines.append(
            f"| {row['fold']} | {_fmt(row['unrestricted'])} | {_fmt(row['depth_8'])} | "
            f"{_fmt(row['pruned'])} | {_fmt(row['baseline_forest'])} | {_fmt(row['tuned_forest'])} |"
        )
    class_lines = [
        "| Model | Class | Precision | Recall | F1 | Support |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["per_class"]:
        class_lines.append(
            f"| {row['model']} | {row['class']} | {_fmt(row['precision'])} | "
            f"{_fmt(row['recall'])} | {_fmt(row['f1'])} | {row['support']} |"
        )
    matrix_header = "| True \\ Predicted | " + " | ".join(CLASS_ORDER) + " |"
    matrix_sep = "| --- | " + " | ".join(["---:"] * len(CLASS_ORDER)) + " |"
    matrix_lines = [matrix_header, matrix_sep]
    for label, row in zip(CLASS_ORDER, payload["confusion_matrix"]):
        matrix_lines.append("| " + label + " | " + " | ".join(str(int(value)) for value in row) + " |")
    text = "\n".join(
        [
            "# Stage 4: Random forest baseline, tuning, and comparison",
            "",
            "The feature list is still provisional. This stage uses the same 76 primary features, "
            "the 7,381 development rows with a measured transit depth, and the saved `dev_fold` "
            "assignments. The median imputer is fit inside each training fold. The test partition "
            "is not scored. The experimental significance-statistic configuration is not trained. "
            "No result is declared statistically significant.",
            "",
            "## Sample and search",
            "",
            f"- Development rows: {payload['n_rows']:,}",
            f"- Development host stars: {payload['n_groups']:,}",
            f"- Primary features: {payload['n_features']}",
            f"- Saved split SHA-256: `{payload['split_sha256']}`",
            f"- Seed: {payload['random_seed']}",
            f"- Full listed grid: {payload['n_full_grid']} forests. Evaluated: {payload['n_evaluated']}.",
            f"- Baseline, pre-specified: `n_estimators` 100, 200, and 400 with `max_depth=None`, "
            "`max_features='sqrt'`, `min_samples_leaf=1`, and `class_weight=None`.",
            f"- Random phase: {payload['n_random']} configurations drawn with seed {payload['random_seed']} "
            "from the other 100-tree cells, before any validation score was used.",
            f"- Promotion: the {payload['top_promote']} random-phase forests with the highest mean "
            "validation macro-F1 were repeated at 200 and 400 trees.",
            f"- `n_jobs=-1` on {payload['cpu_count']} CPUs. Fit seconds are the sum of the five "
            "training-fold fits and depend on this machine.",
            f"- Total training-fold fit time: {payload['total_fit_seconds']:.1f} seconds. "
            f"Trees grown in those fits: {payload['trees_grown']:,}.",
            "",
            "Selection uses the maximum mean grouped validation macro-F1. Exact ties break toward "
            "the smaller macro-F1 gap, then fewer trees. The tuned score is the score used to "
            "choose the forest. It is not a nested estimate on observations held out from the search. "
            "The three trees were refit on these same folds so precision and recall use the same "
            "metric definition. Their macro-F1 values were checked against the saved Stage 2 and "
            "Stage 3 results.",
            "",
            "## Tuned forest",
            "",
            forest_label(payload["tuned"]),
            "",
            payload["tuned_identity"],
            "",
            "A forest with these hyperparameters was also fit on all development rows for inspection. "
            "That fit is in-sample. It was not scored on the test partition.",
            "",
            f"- In-sample training accuracy: {_fmt(payload['development_fit']['train_accuracy'])}",
            f"- In-sample training macro-F1: {_fmt(payload['development_fit']['train_macro_f1'])}",
            f"- In-sample fit seconds: {payload['development_fit']['fit_seconds']:.1f}",
            "",
            "## Comparison",
            "",
            "Validation metrics are means across the five grouped folds, with the sample standard "
            "deviation. The macro-F1 gap is mean training macro-F1 minus mean validation macro-F1. "
            "Fit seconds are the total of the five training-fold fits.",
            "",
            *comparison,
            "",
            "### Fold-level validation macro-F1",
            "",
            *fold_lines,
            "",
            payload["agreement_pruned"],
            "",
            payload["agreement_baseline"],
            "",
            "## Tuning summary",
            "",
            "The five forests with the highest mean grouped validation macro-F1. "
            "The full evaluated set is `results/stage4/rf_search_summary.csv`.",
            "",
            *tuning,
            "",
            "## Out-of-fold class metrics",
            "",
            "These precision, recall, and F1 values are computed from pooled out-of-fold predictions. "
            "They are not the mean of the five fold scores. The comparison table above remains the "
            "selection summary.",
            "",
            *class_lines,
            "",
            "### Tuned forest confusion matrix",
            "",
            *matrix_lines,
            "",
            f"Pooled out-of-fold macro-F1 for the tuned forest: {_fmt(payload['tuned_pooled_macro_f1'])}.",
            "",
            "## Overfitting comparison",
            "",
            payload["gap_statement"],
            "",
            _selection_tradeoff(payload["comparison"], payload["per_class"], payload["top_forests"]),
            "",
            "## Not done in this stage",
            "",
            "The test partition was not scored. Permutation importance was not computed. "
            "The significance-statistic configuration was not trained. The 363 incomplete KOIs "
            "were not fit. The search did not evaluate every listed forest combination.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _load_pruned_tree() -> dict:
    config = json.loads((STAGE3_RESULTS / "model_config.json").read_text(encoding="utf-8"))
    selected = config["selected_hyperparameters"]
    params = {
        "max_depth": selected["max_depth"],
        "min_samples_split": int(selected["min_samples_split"]),
        "min_samples_leaf": int(selected["min_samples_leaf"]),
        "ccp_alpha": float(selected["ccp_alpha"]),
    }
    if params["ccp_alpha"] <= 0:
        raise RuntimeError("The Stage 3 selected tree has no cost-complexity penalty.")
    return params


def _assert_saved_trees(unrestricted: dict, depth8: dict, pruned: dict) -> None:
    saved = pd.read_csv(STAGE3_RESULTS / "comparison_folds.csv")
    expected = {
        "unrestricted": unrestricted,
        "depth_8": depth8,
        "selected": pruned,
    }
    for name, result in expected.items():
        rows = saved.loc[saved["model"].eq(name)].sort_values("fold")
        for saved_row, fold in zip(rows.to_dict(orient="records"), result["folds"], strict=True):
            for metric in ("train_macro_f1", "validation_macro_f1", "validation_accuracy"):
                if not np.isclose(fold[metric], saved_row[metric], atol=1e-10):
                    raise RuntimeError(f"{name} fold {fold['fold']} {metric} does not match Stage 3.")


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _public_forest(row: dict) -> dict:
    public = {
        key: value
        for key, value in row.items()
        if key not in {"folds", "oof_prediction"}
    }
    public["max_depth"] = "None" if row["max_depth"] is None else int(row["max_depth"])
    public["class_weight"] = "None" if row["class_weight"] is None else row["class_weight"]
    public["max_features"] = (
        row["max_features"] if isinstance(row["max_features"], str) else float(row["max_features"])
    )
    return public


def main() -> None:
    started = time.perf_counter()
    STAGE4_RESULTS.mkdir(parents=True, exist_ok=True)
    STAGE4_FIGURES.mkdir(parents=True, exist_ok=True)
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

    def evaluate_tree(params: dict) -> dict:
        print(f"Evaluating tree ccp_alpha={params['ccp_alpha']} max_depth={params['max_depth']}", flush=True)
        result = cross_validate_pipeline(
            development,
            columns,
            lambda params=params: make_tree_pipeline(
                params["max_depth"],
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                ccp_alpha=float(params["ccp_alpha"]),
            ),
        )
        result.update(params)
        return result

    unrestricted = evaluate_tree(UNRESTRICTED_TREE)
    depth8 = evaluate_tree(DEPTH8_TREE)
    pruned_params = _load_pruned_tree()
    pruned = evaluate_tree(pruned_params)
    _assert_saved_trees(unrestricted, depth8, pruned)

    cache: dict[tuple, dict] = {}

    def evaluate_forest(params: dict) -> dict:
        key = forest_key(params)
        if key not in cache:
            print(f"Evaluating forest {forest_label(params)}", flush=True)
            result = cross_validate_pipeline(
                development,
                columns,
                lambda params=params: make_forest_pipeline(params),
            )
            result.update(params)
            cache[key] = result
            print(
                f"  validation macro-F1 {result['mean_validation_macro_f1']:.4f} "
                f"in {result['total_fit_seconds']:.1f}s",
                flush=True,
            )
        return cache[key]

    for n_estimators in N_ESTIMATORS_GRID:
        baseline = dict(BASELINE)
        baseline["n_estimators"] = n_estimators
        evaluate_forest(baseline)
    sampled = random_phase()
    for params in sampled:
        evaluate_forest(params)
    phase_rows = [cache[forest_key(params)] for params in sampled]
    ranked = sorted(phase_rows, key=lambda row: row["mean_validation_macro_f1"], reverse=True)
    for row in ranked[:TOP_PROMOTE]:
        for n_estimators in (200, 400):
            promoted = {
                "n_estimators": n_estimators,
                "max_depth": row["max_depth"],
                "max_features": row["max_features"],
                "min_samples_leaf": int(row["min_samples_leaf"]),
                "class_weight": row["class_weight"],
            }
            evaluate_forest(promoted)

    forests = list(cache.values())
    tuned = select_forest(forests)
    baseline_result = cache[forest_key(BASELINE)]
    tuned_identity = (
        "The search selected the baseline forest."
        if forest_key(tuned) == forest_key(BASELINE)
        else "The search selected a different forest from the 100-tree baseline."
    )

    agreement_pruned = fold_agreement(
        [
            tuned_fold["validation_macro_f1"] - pruned_fold["validation_macro_f1"]
            for tuned_fold, pruned_fold in zip(tuned["folds"], pruned["folds"], strict=True)
        ]
    )
    agreement_baseline = fold_agreement(
        [
            tuned_fold["validation_macro_f1"] - baseline_fold["validation_macro_f1"]
            for tuned_fold, baseline_fold in zip(tuned["folds"], baseline_result["folds"], strict=True)
        ]
    )
    fold_table = []
    for index in range(5):
        fold_table.append(
            {
                "fold": index,
                "unrestricted": unrestricted["folds"][index]["validation_macro_f1"],
                "depth_8": depth8["folds"][index]["validation_macro_f1"],
                "pruned": pruned["folds"][index]["validation_macro_f1"],
                "baseline_forest": baseline_result["folds"][index]["validation_macro_f1"],
                "tuned_forest": tuned["folds"][index]["validation_macro_f1"],
            }
        )

    in_sample = make_forest_pipeline(tuned)
    fit_started = time.perf_counter()
    in_sample.fit(development[columns], development[TARGET_COLUMN])
    in_sample_seconds = time.perf_counter() - fit_started
    in_sample_scores = _classification_scores(
        development[TARGET_COLUMN],
        in_sample.predict(development[columns]),
    )
    development_fit = {
        "train_accuracy": in_sample_scores["accuracy"],
        "train_macro_f1": in_sample_scores["macro_f1"],
        "fit_seconds": float(in_sample_seconds),
        "scored_on_test": False,
        "role": "in_sample_development_fit",
    }

    def role(name: str, row: dict) -> dict:
        return {"role": name, **row}

    comparison = [
        role("Unrestricted tree", unrestricted),
        role("Depth-8 tree", depth8),
        role("Pruned tree", pruned),
        role("Baseline forest", baseline_result),
        role("Tuned forest", tuned),
    ]
    per_class = []
    for model in comparison:
        for class_row in _per_class(development[TARGET_COLUMN], model["oof_prediction"]):
            per_class.append({"model": model["role"], **class_row})
    matrix = confusion_matrix(
        development[TARGET_COLUMN],
        tuned["oof_prediction"],
        labels=CLASS_ORDER,
    )
    tuned_pooled = _classification_scores(development[TARGET_COLUMN], tuned["oof_prediction"])
    top_forests = sorted(
        forests,
        key=lambda row: (row["mean_validation_macro_f1"], -row["mean_macro_f1_gap"]),
        reverse=True,
    )[:5]
    trees_grown = int(sum(int(row["n_estimators"]) * 5 for row in forests))
    total_fit_seconds = float(sum(row["total_fit_seconds"] for row in forests))
    payload = {
        "stage": 4,
        "team_finalized": False,
        "random_seed": RANDOM_SEED,
        "split_sha256": digest,
        "partition": "development",
        "test_scored": False,
        "n_rows": int(len(development)),
        "n_groups": int(development[GROUP_COLUMN].nunique()),
        "n_features": len(columns),
        "n_full_grid": len(forest_grid()),
        "n_evaluated": len(forests),
        "n_random": N_RANDOM,
        "top_promote": TOP_PROMOTE,
        "cpu_count": int(os.cpu_count() or 1),
        "total_fit_seconds": total_fit_seconds,
        "trees_grown": trees_grown,
        "wall_seconds": time.perf_counter() - started,
        "tuned": tuned,
        "tuned_identity": tuned_identity,
        "comparison": comparison,
        "top_forests": top_forests,
        "fold_macro_f1": fold_table,
        "agreement_pruned": _agreement_sentence("pruned tree", agreement_pruned),
        "agreement_baseline": _agreement_sentence("baseline forest", agreement_baseline),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "tuned_pooled_macro_f1": tuned_pooled["macro_f1"],
        "gap_statement": _gap_statement(tuned, pruned, depth8, unrestricted, baseline_result),
        "development_fit": development_fit,
        "random_sample": sampled,
    }
    pd.DataFrame([_public_forest(row) for row in forests]).sort_values(
        ["mean_validation_macro_f1", "mean_macro_f1_gap"],
        ascending=[False, True],
    ).to_csv(STAGE4_RESULTS / "rf_search_summary.csv", index=False)
    comparison_public = []
    for model in comparison:
        public = _public_forest(model) if "n_estimators" in model else {
            key: value
            for key, value in model.items()
            if key not in {"folds", "oof_prediction"}
        }
        public["role"] = model["role"]
        comparison_public.append(public)
    pd.DataFrame(comparison_public).to_csv(STAGE4_RESULTS / "comparison.csv", index=False)
    fold_records = []
    for model in comparison:
        for fold in model["folds"]:
            fold_records.append({"model": model["role"], **fold})
    pd.DataFrame(fold_records).to_csv(STAGE4_RESULTS / "comparison_folds.csv", index=False)
    pd.DataFrame(per_class).to_csv(STAGE4_RESULTS / "per_class_metrics.csv", index=False)
    pd.DataFrame(matrix, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        STAGE4_RESULTS / "oof_confusion_matrix.csv"
    )
    config = {
        "team_finalized": False,
        "feature_configuration": "primary_leakage_aware_provisional",
        "n_features": len(columns),
        "feature_columns": columns,
        "split_file": "data/processed/proposed_splits.csv",
        "split_sha256": digest,
        "test_scored": False,
        "random_seed": RANDOM_SEED,
        "n_jobs": -1,
        "cpu_count": payload["cpu_count"],
        "baseline": BASELINE,
        "random_sample": sampled,
        "selection_rule": (
            "maximum mean grouped validation macro-F1; "
            "exact ties break toward the smaller macro-F1 gap, then fewer trees"
        ),
        "selected_hyperparameters": {
            "n_estimators": int(tuned["n_estimators"]),
            "criterion": "gini",
            "max_depth": tuned["max_depth"],
            "max_features": tuned["max_features"],
            "min_samples_leaf": int(tuned["min_samples_leaf"]),
            "min_samples_split": 2,
            "class_weight": tuned["class_weight"],
            "bootstrap": True,
            "random_state": RANDOM_SEED,
            "imputer": "median",
        },
        "mean_validation_macro_f1": tuned["mean_validation_macro_f1"],
        "mean_macro_f1_gap": tuned["mean_macro_f1_gap"],
        "total_fit_seconds": tuned["total_fit_seconds"],
    }
    _json(STAGE4_RESULTS / "model_config.json", config)
    stored = {
        key: value
        for key, value in payload.items()
        if key not in {"comparison", "tuned", "top_forests", "random_sample"}
    }
    stored["comparison"] = [
        {"role": model["role"], **_summary_only(model)}
        for model in comparison
    ]
    stored["tuned_hyperparameters"] = config["selected_hyperparameters"]
    stored["top_forests"] = [_summary_only(model) for model in top_forests]
    _json(STAGE4_RESULTS / "stage4_metrics.json", stored)
    joblib.dump(
        {
            "pipeline": in_sample,
            "feature_columns": columns,
            "hyperparameters": config["selected_hyperparameters"],
            "role": "in_sample_development_fit",
            "test_scored": False,
            "random_seed": RANDOM_SEED,
        },
        STAGE4_RESULTS / "tuned_forest.joblib",
    )
    _plot_comparison(comparison, STAGE4_FIGURES / "model_comparison.png")
    _plot_confusion(matrix, STAGE4_FIGURES / "oof_confusion_matrix.png")
    _write_findings(payload, DOCS / "stage4_findings.md")
    print(
        "Stage 4 complete: "
        f"{forest_label(tuned)}; "
        f"validation macro-F1 {tuned['mean_validation_macro_f1']:.4f}; "
        f"gap {tuned['mean_macro_f1_gap']:.4f}; "
        f"fit {tuned['total_fit_seconds']:.1f}s."
    )
    print(tuned_identity)
    print(payload["gap_statement"])
    print(payload["agreement_pruned"])
    print(payload["agreement_baseline"])


def _summary_only(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in {"folds", "oof_prediction", "role"}
    }


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
