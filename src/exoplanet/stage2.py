"""Unrestricted Decision Tree on the development partition only.

The saved grouped split is reused. The test partition is not scored.
koi_fwm_stat_sig is not a training feature. No forest is trained.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import PredefinedSplit, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from exoplanet.config import (
    CLASS_ORDER,
    DATA_PROCESSED,
    DOCS,
    GROUP_COLUMN,
    ID_COLUMN,
    RANDOM_SEED,
    ROOT,
    SAVED_SPLIT_SHA256,
    TARGET_COLUMN,
)

STAGE2_RESULTS = ROOT / "results" / "person3" / "decision_tree" / "stage2"
STAGE2_FIGURES = ROOT / "figures" / "person3" / "decision_tree" / "stage2"
LEAKAGE_FORBIDDEN = {
    "koi_pdisposition",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "kepler_name",
    "koi_comment",
    "koi_fwm_stat_sig",
    "kepid",
    TARGET_COLUMN,
}


def make_tree_pipeline(
    max_depth: int | None,
    *,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    ccp_alpha: float = 0.0,
) -> Pipeline:
    """Median imputation plus a Decision Tree.

    The defaults leave the tree unpruned apart from an optional depth limit.
    max_depth None with those defaults is the unrestricted tree.
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "tree",
                DecisionTreeClassifier(
                    criterion="gini",
                    splitter="best",
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    min_weight_fraction_leaf=0.0,
                    max_features=None,
                    max_leaf_nodes=None,
                    min_impurity_decrease=0.0,
                    ccp_alpha=ccp_alpha,
                    class_weight=None,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def development_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Return development rows and refuse any other partition."""
    if "partition" not in frame.columns:
        raise RuntimeError("The modeling frame has no partition column.")
    if frame["partition"].eq("test").any() or frame["partition"].ne("development").any():
        raise RuntimeError(
            "The modeling frame includes rows outside the development partition. "
            "The test partition is not scored."
        )
    if frame.empty:
        raise RuntimeError("The development modeling frame is empty.")
    return frame


def assert_primary_features(columns: list[str]) -> None:
    forbidden = LEAKAGE_FORBIDDEN & set(columns)
    if forbidden:
        raise RuntimeError(f"Primary features include forbidden columns: {sorted(forbidden)}")
    if len(columns) != len(set(columns)):
        raise RuntimeError("The primary feature list contains duplicates.")


def _scores(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER)),
    }


def evaluate_unrestricted(frame: pd.DataFrame, feature_columns: list[str]) -> dict:
    """Grouped fold scores, depth, leaves, and out-of-fold predictions."""
    frame = development_only(frame)
    assert_primary_features(feature_columns)
    if frame.groupby(GROUP_COLUMN)["dev_fold"].nunique().max() != 1:
        raise RuntimeError("A host star appears in more than one development fold.")
    folds = frame["dev_fold"].to_numpy()
    if set(np.unique(folds)) != set(range(5)):
        raise RuntimeError(f"Expected development folds 0-4, found {sorted(set(folds))}.")
    x = frame[feature_columns]
    y = frame[TARGET_COLUMN]
    oof = pd.Series(index=frame.index, dtype=object)
    fold_rows = []
    for fold_id in range(5):
        train = folds != fold_id
        valid = folds == fold_id
        if frame.loc[train, GROUP_COLUMN].isin(frame.loc[valid, GROUP_COLUMN]).any():
            raise RuntimeError(f"Fold {fold_id} shares a host star with its training rows.")
        for part in (train, valid):
            present = set(y.loc[part].unique())
            if present != set(CLASS_ORDER):
                raise RuntimeError(f"Fold {fold_id} is missing a class: {sorted(present)}")
        pipeline = make_tree_pipeline(None)
        pipeline.fit(x.loc[train], y.loc[train])
        tree = pipeline.named_steps["tree"]
        pred_train = pipeline.predict(x.loc[train])
        pred_valid = pipeline.predict(x.loc[valid])
        oof.loc[frame.index[valid]] = pred_valid
        train_scores = _scores(y.loc[train], pred_train)
        valid_scores = _scores(y.loc[valid], pred_valid)
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": int(train.sum()),
                "n_validation": int(valid.sum()),
                "n_train_groups": int(frame.loc[train, GROUP_COLUMN].nunique()),
                "n_validation_groups": int(frame.loc[valid, GROUP_COLUMN].nunique()),
                "train_accuracy": train_scores["accuracy"],
                "train_macro_f1": train_scores["macro_f1"],
                "validation_accuracy": valid_scores["accuracy"],
                "validation_macro_f1": valid_scores["macro_f1"],
                "depth": int(tree.get_depth()),
                "n_leaves": int(tree.get_n_leaves()),
            }
        )
    if oof.isna().any():
        raise RuntimeError("An out-of-fold prediction is missing.")
    matrix = confusion_matrix(y, oof, labels=CLASS_ORDER)
    return {
        "folds": fold_rows,
        "oof_prediction": oof,
        "confusion_matrix": matrix,
    }


def depth_grid(observed_depths: list[int]) -> list[int]:
    deepest = max(observed_depths)
    anchors = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100]
    chosen = [depth for depth in anchors if depth < deepest]
    if deepest not in chosen:
        chosen.append(deepest)
    return chosen


def run_validation_curve(
    frame: pd.DataFrame,
    feature_columns: list[str],
    finite_depths: list[int],
) -> pd.DataFrame:
    """Training and grouped validation scores across max_depth."""
    frame = development_only(frame)
    assert_primary_features(feature_columns)
    param_range: list[int | None] = list(finite_depths) + [None]
    cv = PredefinedSplit(frame["dev_fold"].to_numpy())
    x = frame[feature_columns]
    y = frame[TARGET_COLUMN]
    rows = []
    score_sets = {}
    for metric, scoring in (("accuracy", "accuracy"), ("macro_f1", "f1_macro")):
        train_scores, valid_scores = validation_curve(
            make_tree_pipeline(None),
            x,
            y,
            param_name="tree__max_depth",
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        score_sets[metric] = (train_scores, valid_scores)
    for index, depth in enumerate(param_range):
        row = {
            "max_depth": depth,
            "unrestricted": depth is None,
        }
        for metric, (train_scores, valid_scores) in score_sets.items():
            row[f"mean_train_{metric}"] = float(train_scores[index].mean())
            row[f"std_train_{metric}"] = float(train_scores[index].std(ddof=1))
            row[f"mean_validation_{metric}"] = float(valid_scores[index].mean())
            row[f"std_validation_{metric}"] = float(valid_scores[index].std(ddof=1))
            row[f"train_{metric}_folds"] = [float(value) for value in train_scores[index]]
            row[f"validation_{metric}_folds"] = [float(value) for value in valid_scores[index]]
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def curve_statement(curve: pd.DataFrame, unrestricted: dict[str, float]) -> str:
    """Describe the measured train–validation pattern without a preset threshold."""
    finite = curve.loc[~curve["unrestricted"]].copy()
    best = finite.loc[finite["mean_validation_macro_f1"].idxmax()]
    shallow = finite.loc[finite["max_depth"].idxmin()]
    train = unrestricted["mean_train_macro_f1"]
    valid = unrestricted["mean_validation_macro_f1"]
    gap = train - valid
    train_gap = train - float(best["mean_train_macro_f1"])
    valid_gap = float(best["mean_validation_macro_f1"]) - valid
    sentences = [
        (
            f"The unrestricted tree has mean training macro-F1 {train:.4f} and "
            f"mean grouped validation macro-F1 {valid:.4f}. "
            f"Training minus validation is {gap:.4f}."
        ),
        (
            f"At max_depth={int(shallow['max_depth'])}, mean training macro-F1 is "
            f"{float(shallow['mean_train_macro_f1']):.4f} and mean validation macro-F1 is "
            f"{float(shallow['mean_validation_macro_f1']):.4f}."
        ),
        (
            f"On this depth grid, the highest mean validation macro-F1 is "
            f"{float(best['mean_validation_macro_f1']):.4f} at max_depth={int(best['max_depth'])}, "
            f"where mean training macro-F1 is {float(best['mean_train_macro_f1']):.4f}."
        ),
    ]
    if gap > 0 and train_gap > 0 and valid_gap >= 0:
        sentences.append(
            "Training macro-F1 is higher for the unrestricted tree than for the depth "
            f"with the highest grouped validation macro-F1, by {train_gap:.4f}. "
            "The unrestricted grouped validation macro-F1 is "
            + (
                f"{valid_gap:.4f} lower than that depth. "
                if valid_gap > 0
                else "the same as that depth. "
            )
            + "The extra depth improves the fit to the training folds and does not "
            "improve the grouped validation score. That is the overfitting shown by "
            "this development partition."
        )
    elif gap > 0:
        sentences.append(
            "The unrestricted tree scores higher on its training folds than on the "
            "held-out development folds. The depth curve does not show a lower "
            "validation score at the unrestricted tree than at the best shallower "
            "depth, so the result is the measured gap above rather than a peak-and-decline pattern."
        )
    else:
        sentences.append(
            "The unrestricted tree's mean training macro-F1 does not exceed its mean "
            "grouped validation macro-F1, so these scores do not show a training-over-validation gap."
        )
    return " ".join(sentences)


def _load_inputs() -> tuple[pd.DataFrame, list[str], str]:
    split_path = DATA_PROCESSED / "proposed_splits.csv"
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if digest != SAVED_SPLIT_SHA256:
        raise RuntimeError("Stage 2 refused to run because the saved split file changed.")
    features = pd.read_csv(DATA_PROCESSED / "proposed_leakage_aware_features.csv")
    columns = features["column"].astype(str).tolist()
    assert_primary_features(columns)
    experimental = pd.read_csv(DATA_PROCESSED / "experimental_features_with_fwm_stat_sig.csv")
    extra = experimental.loc[experimental["role"].eq("experimental_only"), "column"].tolist()
    if extra != ["koi_fwm_stat_sig"]:
        raise RuntimeError("The experimental configuration is not limited to koi_fwm_stat_sig.")
    if "koi_fwm_stat_sig" in columns:
        raise RuntimeError("koi_fwm_stat_sig is in the primary feature file.")
    matrix = pd.read_csv(DATA_PROCESSED / "proposed_primary_matrix_unimputed.csv")
    missing = [column for column in columns if column not in matrix.columns]
    if missing:
        raise RuntimeError(f"Primary matrix is missing features: {missing}")
    if "koi_fwm_stat_sig" in matrix.columns:
        raise RuntimeError("The primary matrix still contains koi_fwm_stat_sig.")
    splits = pd.read_csv(split_path)
    development = matrix.loc[matrix["partition"].eq("development")].copy()
    test_ids = set(splits.loc[splits["partition"].eq("test"), ID_COLUMN].astype(str))
    if set(development[ID_COLUMN].astype(str)) & test_ids:
        raise RuntimeError("A test KOI entered the development modeling frame.")
    development_only(development)
    return development, columns, digest


def _plot_curve(curve: pd.DataFrame, mean_depth: float, path: Path) -> None:
    finite = curve.loc[~curve["unrestricted"]]
    unrestricted = curve.loc[curve["unrestricted"]].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    specs = (
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
    )
    for axis, (metric, label) in zip(axes, specs):
        x = finite["max_depth"].to_numpy()
        axis.errorbar(
            x,
            finite[f"mean_train_{metric}"],
            yerr=finite[f"std_train_{metric}"],
            marker="o",
            label="Training folds",
            color="#0072B2",
        )
        axis.errorbar(
            x,
            finite[f"mean_validation_{metric}"],
            yerr=finite[f"std_validation_{metric}"],
            marker="o",
            label="Grouped validation folds",
            color="#D55E00",
        )
        axis.scatter(
            [mean_depth],
            [unrestricted[f"mean_validation_{metric}"]],
            marker="*",
            s=140,
            color="#D55E00",
            zorder=3,
            label="Unrestricted validation",
        )
        axis.scatter(
            [mean_depth],
            [unrestricted[f"mean_train_{metric}"]],
            marker="*",
            s=140,
            color="#0072B2",
            zorder=3,
            label="Unrestricted training",
        )
        axis.set_xlabel("max_depth")
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.set_ylim(0, 1.05)
        axis.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Decision Tree validation curve on the development partition")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
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
    axis.set_title("Development out-of-fold predictions, unrestricted tree")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _write_findings(payload: dict, path: Path) -> None:
    folds = payload["folds"]
    summary = payload["summary"]
    curve_lines = [
        "| max_depth | Train accuracy | Validation accuracy | Train macro-F1 | Validation macro-F1 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["curve"]:
        depth = "unrestricted" if row["unrestricted"] else str(int(row["max_depth"]))
        curve_lines.append(
            f"| {depth} | {_fmt(row['mean_train_accuracy'])} ± {_fmt(row['std_train_accuracy'])} "
            f"| {_fmt(row['mean_validation_accuracy'])} ± {_fmt(row['std_validation_accuracy'])} "
            f"| {_fmt(row['mean_train_macro_f1'])} ± {_fmt(row['std_train_macro_f1'])} "
            f"| {_fmt(row['mean_validation_macro_f1'])} ± {_fmt(row['std_validation_macro_f1'])} |"
        )
    fold_lines = [
        "| Fold | Train rows | Validation rows | Train accuracy | Validation accuracy | Train macro-F1 | Validation macro-F1 | Depth | Leaves |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in folds:
        fold_lines.append(
            f"| {row['fold']} | {row['n_train']} | {row['n_validation']} "
            f"| {_fmt(row['train_accuracy'])} | {_fmt(row['validation_accuracy'])} "
            f"| {_fmt(row['train_macro_f1'])} | {_fmt(row['validation_macro_f1'])} "
            f"| {row['depth']} | {row['n_leaves']} |"
        )
    labels = payload["class_order"]
    header = "| True \\ Predicted | " + " | ".join(labels) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(labels)) + " |"
    matrix_lines = [header, sep]
    for label, row in zip(labels, payload["confusion_matrix"]):
        matrix_lines.append(f"| {label} | " + " | ".join(str(int(value)) for value in row) + " |")
    text = "\n".join(
        [
            "# Stage 2: unrestricted Decision Tree",
            "",
            "The feature list is still provisional. This stage uses the primary leakage-aware "
            "features and the development rows that have a measured transit depth. "
            "The 363 incomplete KOIs stay in the saved split for a later sensitivity experiment "
            "and are not used here. `koi_fwm_stat_sig` is not a training feature. "
            "The experimental configuration that adds it is not trained. "
            "The test partition is not scored. No forest is trained. No depth is selected as a final model.",
            "",
            "## Sample",
            "",
            f"- Development rows: {payload['n_rows']:,}",
            f"- Development host stars: {payload['n_groups']:,}",
            f"- Primary features: {payload['n_features']}",
            f"- Saved split SHA-256: `{payload['split_sha256']}`",
            f"- Seed: {payload['random_seed']}",
            "",
            "## Model",
            "",
            "Each fold fits a median imputer on its training rows, then a "
            "`DecisionTreeClassifier` with `criterion='gini'`, `max_depth=None`, "
            "`min_samples_split=2`, `min_samples_leaf=1`, `max_leaf_nodes=None`, "
            "`ccp_alpha=0.0`, `class_weight=None`, and `random_state=245`. "
            "The validation curve changes only `max_depth`.",
            "",
            "## Grouped cross-validation",
            "",
            "Folds are the `dev_fold` values stored in `data/processed/proposed_splits.csv`. "
            "They were not recomputed.",
            "",
            *fold_lines,
            "",
            f"Mean training accuracy {_fmt(summary['train_accuracy']['mean'])} "
            f"(std {_fmt(summary['train_accuracy']['std'])}). "
            f"Mean grouped validation accuracy {_fmt(summary['validation_accuracy']['mean'])} "
            f"(std {_fmt(summary['validation_accuracy']['std'])}).",
            "",
            f"Mean training macro-F1 {_fmt(summary['train_macro_f1']['mean'])} "
            f"(std {_fmt(summary['train_macro_f1']['std'])}). "
            f"Mean grouped validation macro-F1 {_fmt(summary['validation_macro_f1']['mean'])} "
            f"(std {_fmt(summary['validation_macro_f1']['std'])}).",
            "",
            f"Depth by fold: {', '.join(str(row['depth']) for row in folds)} "
            f"(mean {summary['depth']['mean']:.2f}). "
            f"Leaves by fold: {', '.join(str(row['n_leaves']) for row in folds)} "
            f"(mean {summary['n_leaves']['mean']:.1f}).",
            "",
            "## In-sample fit on all development rows",
            "",
            "This fit is saved so the unrestricted tree can be inspected. Its scores are "
            "training scores on the same rows used to grow the tree. They are not a "
            "validation estimate, and this model was not applied to the test partition.",
            "",
            f"- Training accuracy: {_fmt(payload['development_fit']['train_accuracy'])}",
            f"- Training macro-F1: {_fmt(payload['development_fit']['train_macro_f1'])}",
            f"- Depth: {payload['development_fit']['depth']}",
            f"- Leaves: {payload['development_fit']['n_leaves']}",
            "",
            "## Validation curve",
            "",
            "Values are the mean across the five grouped folds, plus the sample standard deviation. "
            "The unrestricted point uses `max_depth=None`.",
            "",
            *curve_lines,
            "",
            payload["curve_statement"],
            "",
            "## Out-of-fold confusion matrix",
            "",
            "Each development row is predicted by the unrestricted tree trained on the other four folds.",
            "",
            *matrix_lines,
            "",
            "## Not done in this stage",
            "",
            "No pruned depth was chosen. The experimental feature set that adds "
            "`koi_fwm_stat_sig` was not trained. The 363 incomplete KOIs were not fit. "
            "The test partition was not scored. A forest was not trained.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    STAGE2_RESULTS.mkdir(parents=True, exist_ok=True)
    STAGE2_FIGURES.mkdir(parents=True, exist_ok=True)
    development, columns, digest = _load_inputs()
    evaluated = evaluate_unrestricted(development, columns)
    folds = evaluated["folds"]
    summary = {
        "train_accuracy": _mean_std([row["train_accuracy"] for row in folds]),
        "validation_accuracy": _mean_std([row["validation_accuracy"] for row in folds]),
        "train_macro_f1": _mean_std([row["train_macro_f1"] for row in folds]),
        "validation_macro_f1": _mean_std([row["validation_macro_f1"] for row in folds]),
        "depth": _mean_std([row["depth"] for row in folds]),
        "n_leaves": _mean_std([row["n_leaves"] for row in folds]),
    }
    finite_depths = depth_grid([row["depth"] for row in folds])
    curve = run_validation_curve(development, columns, finite_depths)
    unrestricted_curve = curve.loc[curve["unrestricted"]].iloc[0]
    for metric in ("accuracy", "macro_f1"):
        manual = np.array([row[f"validation_{metric}"] for row in folds])
        library = np.array(unrestricted_curve[f"validation_{metric}_folds"])
        if not np.allclose(manual, library):
            raise RuntimeError(f"Unrestricted validation {metric} does not match the validation curve.")
        manual_train = np.array([row[f"train_{metric}"] for row in folds])
        library_train = np.array(unrestricted_curve[f"train_{metric}_folds"])
        if not np.allclose(manual_train, library_train):
            raise RuntimeError(f"Unrestricted training {metric} does not match the validation curve.")

    pipeline = make_tree_pipeline(None)
    pipeline.fit(development[columns], development[TARGET_COLUMN])
    in_sample = pipeline.predict(development[columns])
    development_fit = {
        "train_accuracy": _scores(development[TARGET_COLUMN], in_sample)["accuracy"],
        "train_macro_f1": _scores(development[TARGET_COLUMN], in_sample)["macro_f1"],
        "depth": int(pipeline.named_steps["tree"].get_depth()),
        "n_leaves": int(pipeline.named_steps["tree"].get_n_leaves()),
        "scored_on_test": False,
        "role": "in_sample_development_fit",
    }
    statement = curve_statement(
        curve,
        {
            "mean_train_macro_f1": summary["train_macro_f1"]["mean"],
            "mean_validation_macro_f1": summary["validation_macro_f1"]["mean"],
        },
    )
    curve_public = curve.drop(
        columns=[column for column in curve.columns if column.endswith("_folds")]
    )
    payload = {
        "stage": 2,
        "model": "DecisionTreeClassifier",
        "team_finalized": False,
        "random_seed": RANDOM_SEED,
        "split_sha256": digest,
        "partition": "development",
        "test_scored": False,
        "experimental_fwm_stat_sig_trained": False,
        "n_rows": int(len(development)),
        "n_groups": int(development[GROUP_COLUMN].nunique()),
        "n_features": len(columns),
        "feature_columns": columns,
        "class_order": CLASS_ORDER,
        "hyperparameters": {
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_leaf_nodes": None,
            "ccp_alpha": 0.0,
            "class_weight": None,
            "random_state": RANDOM_SEED,
            "imputer": "median",
        },
        "folds": folds,
        "summary": summary,
        "development_fit": development_fit,
        "curve": json.loads(curve.to_json(orient="records")),
        "curve_statement": statement,
        "confusion_matrix": evaluated["confusion_matrix"].tolist(),
    }
    _json(STAGE2_RESULTS / "unrestricted_tree_metrics.json", payload)
    pd.DataFrame(folds).to_csv(STAGE2_RESULTS / "fold_metrics.csv", index=False)
    curve_public.to_csv(STAGE2_RESULTS / "validation_curve.csv", index=False)
    pd.DataFrame(evaluated["confusion_matrix"], index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        STAGE2_RESULTS / "oof_confusion_matrix.csv"
    )
    _json(
        STAGE2_RESULTS / "model_config.json",
        {
            "team_finalized": False,
            "feature_configuration": "primary_leakage_aware_provisional",
            "excluded_from_training": ["koi_fwm_stat_sig"],
            "experimental_configuration_not_trained": "data/processed/experimental_features_with_fwm_stat_sig.csv",
            "split_file": "data/processed/proposed_splits.csv",
            "split_sha256": digest,
            "hyperparameters": payload["hyperparameters"],
            "validation_curve_max_depth": finite_depths + [None],
            "test_scored": False,
        },
    )
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": columns,
            "role": "in_sample_development_fit",
            "test_scored": False,
            "random_seed": RANDOM_SEED,
        },
        STAGE2_RESULTS / "development_tree.joblib",
    )
    mean_depth = summary["depth"]["mean"]
    _plot_curve(curve, mean_depth, STAGE2_FIGURES / "validation_curve.png")
    _plot_confusion(evaluated["confusion_matrix"], STAGE2_FIGURES / "oof_confusion_matrix.png")
    _write_findings(payload, DOCS / "stage2_findings.md")
    print(
        "Stage 2 complete: "
        f"train macro-F1 {summary['train_macro_f1']['mean']:.4f}, "
        f"validation macro-F1 {summary['validation_macro_f1']['mean']:.4f}, "
        f"mean depth {mean_depth:.2f}, mean leaves {summary['n_leaves']['mean']:.1f}."
    )
    print(statement)


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
