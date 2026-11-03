"""Feature importance for the selected tree and forest on development folds.

The saved models are not changed. The test partition is not scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer

from exoplanet.config import (
    CLASS_ORDER,
    DATA_EXTERNAL,
    DATA_PROCESSED,
    DOCS,
    GROUP_COLUMN,
    RANDOM_SEED,
    ROOT,
    TARGET_COLUMN,
)
from exoplanet.stage2 import _load_inputs, development_only, make_tree_pipeline
from exoplanet.stage3 import STAGE3_RESULTS
from exoplanet.stage4 import STAGE4_RESULTS, make_forest_pipeline

STAGE5_RESULTS = ROOT / "results" / "person3" / "feature_importance"
STAGE5_FIGURES = ROOT / "figures" / "person3" / "feature_importance"

N_REPEATS = 10
TOP_N = 15
CORRELATION_THRESHOLD = 0.7
STABILITY_SPEARMAN = 0.8
STABILITY_TOP_N = 10
STABILITY_WORST_RANK = 15
STABILITY_TOP_REQUIRED = 8
CENTROID_TOP_N = 10
CENTROID_TOP_COUNT = 5
CENTROID_POSITIVE_SHARE = 0.5


def macro_f1(y_true, y_pred) -> float:
    return float(
        f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0)
    )


SCORER = make_scorer(macro_f1)


def centroid_columns(feature_table: pd.DataFrame) -> list[str]:
    """Provisional centroid-offset measurements and their uncertainty columns."""
    prefixes = ("koi_fwm_", "koi_dicco_", "koi_dikco_")
    names = []
    for row in feature_table.itertuples(index=False):
        column = str(row.column)
        is_centroid = column.startswith(prefixes) and column != "koi_fwm_stat_sig"
        if is_centroid != bool(row.provisional):
            raise RuntimeError(f"Provisional flag does not match centroid naming for {column}.")
        if is_centroid:
            names.append(column)
    return names


def centroid_dominates(ranked_columns: list[str], centroid: set[str], positive_share: float) -> bool:
    """Pre-specified rule for whether centroid offsets dominate a ranking."""
    top = ranked_columns[:CENTROID_TOP_N]
    count = sum(column in centroid for column in top)
    lead = bool(top) and top[0] in centroid
    return lead or count >= CENTROID_TOP_COUNT or positive_share >= CENTROID_POSITIVE_SHARE


def ranking_is_stable(mean_pairwise_spearman: float, worst_ranks: list[int]) -> bool:
    """Pre-specified rule for whether the leading permutation ranks hold across folds."""
    held = sum(rank <= STABILITY_WORST_RANK for rank in worst_ranks[:STABILITY_TOP_N])
    return mean_pairwise_spearman >= STABILITY_SPEARMAN and held >= STABILITY_TOP_REQUIRED


def _load_selected() -> tuple[dict, dict]:
    tree = json.loads((STAGE3_RESULTS / "model_config.json").read_text(encoding="utf-8"))
    forest = json.loads((STAGE4_RESULTS / "model_config.json").read_text(encoding="utf-8"))
    return tree["selected_hyperparameters"], forest["selected_hyperparameters"]


def _pairwise_spearman(fold_matrix: np.ndarray) -> list[float]:
    values = []
    for left in range(fold_matrix.shape[0]):
        for right in range(left + 1, fold_matrix.shape[0]):
            rho = pd.Series(fold_matrix[left]).corr(pd.Series(fold_matrix[right]), method="spearman")
            values.append(float(rho))
    return values


def _ranks(fold_matrix: np.ndarray) -> np.ndarray:
    """Rank 1 is the largest importance in that fold."""
    order = np.argsort(-fold_matrix, axis=1)
    ranks = np.empty_like(order)
    for fold in range(order.shape[0]):
        ranks[fold, order[fold]] = np.arange(1, order.shape[1] + 1)
    return ranks


def _evaluate_fold(pipeline, x_train, y_train, x_valid, y_valid, fold_id: int) -> dict:
    pipeline.fit(x_train, y_train)
    observed = macro_f1(y_valid, pipeline.predict(x_valid))
    model = pipeline.steps[-1][1]
    impurity = np.asarray(model.feature_importances_, dtype=float)
    permuted = permutation_importance(
        pipeline,
        x_valid,
        y_valid,
        scoring=SCORER,
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED + fold_id,
        n_jobs=1,
    )
    return {
        "validation_macro_f1": observed,
        "impurity": impurity,
        "permutation_mean": np.asarray(permuted.importances_mean, dtype=float),
        "permutation_std": np.asarray(permuted.importances_std, dtype=float),
    }


def _importance_frame(
    columns: list[str],
    fold_matrix: np.ndarray,
    impurity_matrix: np.ndarray | None = None,
    repeat_std: np.ndarray | None = None,
) -> pd.DataFrame:
    ranks = _ranks(fold_matrix)
    frame = pd.DataFrame(
        {
            "column": columns,
            "mean_importance": fold_matrix.mean(axis=0),
            "std_across_folds": fold_matrix.std(axis=0, ddof=1),
            "min_fold_importance": fold_matrix.min(axis=0),
            "max_fold_importance": fold_matrix.max(axis=0),
            "mean_rank": ranks.mean(axis=0),
            "best_fold_rank": ranks.min(axis=0),
            "worst_fold_rank": ranks.max(axis=0),
        }
    )
    if impurity_matrix is not None:
        frame["mean_impurity"] = impurity_matrix.mean(axis=0)
        frame["std_impurity_across_folds"] = impurity_matrix.std(axis=0, ddof=1)
    if repeat_std is not None:
        frame["mean_repeat_std"] = repeat_std.mean(axis=0)
    frame["rank"] = frame["mean_importance"].rank(ascending=False, method="min").astype(int)
    return frame.sort_values(["rank", "column"])


def _positive_share(importance: pd.Series, mask: pd.Series) -> float:
    positive = importance.clip(lower=0)
    total = float(positive.sum())
    if total <= 0:
        return 0.0
    return float(positive.loc[mask].sum() / total)


def _correlation_pairs(development: pd.DataFrame, columns: list[str], top: set[str]) -> pd.DataFrame:
    corr = development[columns].corr(method="spearman")
    rows = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            rho = float(corr.loc[left, right])
            if np.isfinite(rho) and abs(rho) >= CORRELATION_THRESHOLD:
                rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "spearman": rho,
                        "involves_top_feature": left in top or right in top,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=["feature_a", "feature_b", "spearman", "involves_top_feature"]
        )
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)


def _plot_ranked(frame: pd.DataFrame, centroid: set[str], title: str, xlabel: str, path: Path) -> None:
    shown = frame.nsmallest(TOP_N, "rank").iloc[::-1]
    colors = ["#CC79A7" if column in centroid else "#0072B2" for column in shown["column"]]
    fig, axis = plt.subplots(figsize=(8.5, 6.2))
    axis.barh(
        shown["column"],
        shown["mean_importance"],
        xerr=shown["std_across_folds"],
        color=colors,
        capsize=3,
    )
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.legend(
        handles=[
            Patch(facecolor="#0072B2", label="Other feature"),
            Patch(facecolor="#CC79A7", label="Provisional centroid offset"),
        ],
        frameon=False,
        loc="lower right",
    )
    axis.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_comparison(
    forest: pd.DataFrame,
    tree: pd.DataFrame,
    centroid: set[str],
    path: Path,
) -> None:
    forest_top = set(forest.nsmallest(10, "rank")["column"])
    tree_top = set(tree.nsmallest(10, "rank")["column"])
    names = list(forest_top | tree_top)
    forest_idx = forest.set_index("column")
    tree_idx = tree.set_index("column")
    names.sort(key=lambda name: float(forest_idx.loc[name, "mean_importance"]))
    y = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(
        y - 0.18,
        [float(tree_idx.loc[name, "mean_importance"]) for name in names],
        height=0.36,
        color="#009E73",
        label="Pruned tree",
    )
    axis.barh(
        y + 0.18,
        [float(forest_idx.loc[name, "mean_importance"]) for name in names],
        height=0.36,
        color="#0072B2",
        label="Tuned forest",
    )
    labels = [f"{name} *" if name in centroid else name for name in names]
    axis.set_yticks(y, labels)
    axis.set_xlabel("Mean permutation importance (macro-F1 drop)")
    axis.set_title("Development validation folds; * marks a provisional centroid offset")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_rank_paths(fold_matrix: np.ndarray, columns: list[str], order: list[str], path: Path) -> None:
    ranks = _ranks(fold_matrix)
    shown = order[:10]
    fig, axis = plt.subplots(figsize=(8.2, 5.4))
    for name in shown:
        axis.plot(np.arange(1, 6), ranks[:, columns.index(name)], marker="o", label=name)
    axis.set_xticks(np.arange(1, 6), [f"Fold {fold}" for fold in range(5)])
    axis.set_ylabel("Rank (1 = largest macro-F1 drop)")
    axis.set_ylim(0.5, 76.5)
    axis.invert_yaxis()
    axis.set_title("Tuned forest permutation rank on each validation fold")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_correlation(development: pd.DataFrame, columns: list[str], path: Path) -> None:
    if len(columns) < 2:
        return
    corr = development[columns].corr(method="spearman")
    fig, axis = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=axis,
        cbar_kws={"label": "Spearman"},
    )
    axis.set_title("Development rows, Spearman correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _definition_lines(top: pd.DataFrame, dictionary: pd.DataFrame) -> list[str]:
    indexed = dictionary.set_index("column")
    lines = []
    for row in top.itertuples(index=False):
        if row.column not in indexed.index:
            label, description = "not on the parsed column page", "No parsed description is available."
        else:
            info = indexed.loc[row.column]
            label = str(info["label"])
            description = str(info["description"])
        lines.extend(
            [
                f"### `{row.column}`",
                "",
                f"Column-page label: {label}.",
                "",
                description,
                "",
                (
                    f"Mean permutation importance {_fmt(row.mean_importance)} "
                    f"(std across folds {_fmt(row.std_across_folds)}). "
                    f"Fold ranks range from {int(row.best_fold_rank)} to {int(row.worst_fold_rank)}."
                ),
                "",
            ]
        )
    return lines


def _write_findings(payload: dict, path: Path) -> None:
    def table(frame_rows: list[dict]) -> list[str]:
        lines = [
            "| Rank | Column | Mean | Std across folds | Best fold rank | Worst fold rank | Provisional centroid |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in frame_rows:
            lines.append(
                f"| {int(row['rank'])} | `{row['column']}` | {_fmt(row['mean_importance'])} | "
                f"{_fmt(row['std_across_folds'])} | {int(row['best_fold_rank'])} | "
                f"{int(row['worst_fold_rank'])} | {bool(row['centroid'])} |"
            )
        return lines

    pair_lines = [
        "| Feature A | Feature B | Spearman | Involves a top-15 feature |",
        "| --- | --- | ---: | --- |",
    ]
    shown_pairs = [row for row in payload["correlation_pairs"] if row["involves_top_feature"]][:20]
    if not shown_pairs:
        pair_lines.append("|  |  |  | none at the stated threshold |")
    for row in shown_pairs:
        pair_lines.append(
            f"| `{row['feature_a']}` | `{row['feature_b']}` | {_fmt(row['spearman'])} | "
            f"{bool(row['involves_top_feature'])} |"
        )
    stability = payload["stability"]
    centroid = payload["centroid_review"]
    text = "\n".join(
        [
            "# Stage 5: feature importance",
            "",
            "The feature list is still provisional. Importance is computed on the 7,381 "
            "development rows with a measured transit depth and the saved `dev_fold` values. "
            "Each model and its median imputer are fit on that fold's training rows. "
            "Permutation importance is the drop in validation macro-F1 when a raw feature is "
            f"shuffled, repeated {N_REPEATS} times. The test partition is not used. "
            "These results do not change the selected tree or forest.",
            "",
            "## Sample",
            "",
            f"- Development rows: {payload['n_rows']:,}",
            f"- Development host stars: {payload['n_groups']:,}",
            f"- Features: {payload['n_features']}",
            f"- Saved split SHA-256: `{payload['split_sha256']}`",
            f"- Pruned tree: `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, "
            f"`ccp_alpha={payload['tree_ccp_alpha']:.6g}`",
            "- Tuned forest: 100 trees, `max_depth=20`, `max_features='sqrt'`, "
            "`min_samples_leaf=2`, `class_weight='balanced'`",
            "",
            "The validation macro-F1 of these refits matches the Stage 3 and Stage 4 fold scores.",
            "",
            "## Tuned forest, permutation importance",
            "",
            "Values are the mean across the five validation folds. The standard deviation is "
            "across those five fold means, not a hypothesis test. Each fold mean averages "
            f"{N_REPEATS} shuffles. The saved CSV also contains the mean of the within-fold "
            "repeat standard deviations.",
            "",
            *table(payload["forest_top"]),
            "",
            stability["forest_text"],
            "",
            f"Mean pairwise Spearman correlation of the five fold-level importance vectors: "
            f"{_fmt(stability['forest_mean_spearman'])} "
            f"(minimum pair {_fmt(stability['forest_min_spearman'])}).",
            "",
            payload["forest_stability_detail"],
            "",
            "## Impurity importance",
            "",
            "Impurity importance is the mean normalized split improvement on the training folds. "
            "It is a description of the fitted forest, not a held-out score. It can favor features "
            "that offer many similar splits. Its rank correlation with the forest permutation "
            f"importance is {_fmt(payload['impurity_vs_permutation_spearman'])}. "
            "Impurity importance can rank a feature highly when permutation of that feature "
            "on the validation fold does not, especially when a correlated column remains available.",
            "",
            *table(payload["impurity_top"]),
            "",
            "## Pruned tree and tuned forest",
            "",
            f"The Spearman correlation between the two models' mean permutation importances is "
            f"{_fmt(payload['tree_forest_spearman'])}. "
            f"{payload['shared_top10']} of each model's ten leading features are shared. "
            "The absolute drops differ because the pruned tree concentrates its splits, while "
            "the forest can use a correlated column when one feature is shuffled.",
            "",
            "### Pruned tree",
            "",
            *table(payload["tree_top"]),
            "",
            stability["tree_text"],
            "",
            f"Mean pairwise Spearman correlation of the five pruned-tree fold-level importance "
            f"vectors: {_fmt(stability['tree_mean_spearman'])} "
            f"(minimum pair {_fmt(stability['tree_min_spearman'])}).",
            "",
            "## Provisional centroid offsets",
            "",
            centroid["text"],
            "",
            "## Correlated predictors",
            "",
            f"Spearman correlations are computed on the {payload['n_rows']:,} development rows, "
            "without using the test partition and without filling missing values first. "
            f"{payload['n_strong_pairs']} feature pairs have absolute correlation at least "
            f"{CORRELATION_THRESHOLD:.1f}. {payload['n_strong_pairs_in_top']} of them involve "
            "a feature in the forest's top 15. Correlated features can share or split a "
            "permutation drop: shuffling one leaves the other intact, so the measured drop can "
            "understate a redundant signal or move between the pair. A large importance does "
            "not establish that the column causes the disposition.",
            "",
            *pair_lines,
            "",
            "## Column-page definitions for the leading forest features",
            "",
            "The text below is the parsed NASA column-page label and description. "
            "It is not an additional physical conclusion.",
            "",
            *payload["definitions"],
            "",
            "## Limitations",
            "",
            "Permutation importance depends on the fitted model and on the other columns that "
            "remain available. Impurity importance answers a different question and is reported "
            "only as a comparison. The centroid-offset columns are still provisional. "
            "The selected hyperparameters were not revised from these ranks.",
            "",
            "## Not done in this stage",
            "",
            "The test partition was not scored and was not used to compute importance. "
            "No feature was added or removed. The pruned tree and the tuned forest were not replaced.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _stability_text(name: str, stable: bool, held: int, worst: list[int]) -> str:
    shown = ", ".join(str(int(rank)) for rank in worst[:STABILITY_TOP_N])
    if stable:
        return (
            f"The {name} permutation ranking meets the pre-specified stability rule: "
            f"mean pairwise Spearman correlation at least {STABILITY_SPEARMAN:.1f}, and at least "
            f"{STABILITY_TOP_REQUIRED} of the ten leading features have worst-fold rank "
            f"{STABILITY_WORST_RANK} or better. {held} of those ten features meet that rank bound. "
            f"Their worst-fold ranks are {shown}."
        )
    return (
        f"The {name} permutation ranking does not meet the pre-specified stability rule "
        f"(mean pairwise Spearman at least {STABILITY_SPEARMAN:.1f} and at least "
        f"{STABILITY_TOP_REQUIRED} of the ten leading features with worst-fold rank "
        f"{STABILITY_WORST_RANK} or better). {held} of those ten features meet the rank bound. "
        f"Their worst-fold ranks are {shown}."
    )


def _centroid_text(
    model_name: str,
    dominates: bool,
    top_count: int,
    lead: str,
    lead_is_centroid: bool,
    share: float,
) -> str:
    lead_text = (
        f"The leading {model_name} feature is `{lead}`, a provisional centroid-offset column."
        if lead_is_centroid
        else f"The leading {model_name} feature is `{lead}`, which is not a centroid-offset column."
    )
    share_text = (
        f"{top_count} of the top {CENTROID_TOP_N} are provisional centroid-offset columns or "
        f"their uncertainties. Their share of the positive mean permutation importance is {_fmt(share)}."
    )
    if dominates:
        return (
            f"{lead_text} {share_text} Under the pre-specified rule, centroid-offset measurements "
            "dominate this ranking. That is a methodological concern: these columns come from the "
            "vetting analysis and can be strong disposition predictors even though they are not "
            "the archive label. The feature list is still provisional."
        )
    return (
        f"{lead_text} {share_text} Under the pre-specified rule, centroid-offset measurements "
        "do not dominate this ranking. They can still be a methodological concern when they "
        "occupy the next stable ranks: the columns are provisional vetting measurements. "
        "The selected model was not changed."
    )


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


def main() -> None:
    STAGE5_RESULTS.mkdir(parents=True, exist_ok=True)
    STAGE5_FIGURES.mkdir(parents=True, exist_ok=True)
    development, columns, digest = _load_inputs()
    development = development_only(development)
    if len(development) != 7381 or len(columns) != 76:
        raise RuntimeError(
            f"Expected 7381 development rows and 76 features, found {len(development)} and {len(columns)}."
        )
    if development.groupby(GROUP_COLUMN)["dev_fold"].nunique().max() != 1:
        raise RuntimeError("A host star appears in more than one development fold.")
    tree_params, forest_params = _load_selected()
    feature_table = pd.read_csv(DATA_PROCESSED / "proposed_leakage_aware_features.csv")
    if feature_table["column"].astype(str).tolist() != columns:
        raise RuntimeError("The feature file does not match the modeling columns.")
    dictionary = pd.read_csv(DATA_EXTERNAL / "koi_columns_parsed.csv")
    centroids = centroid_columns(feature_table)
    centroid_set = set(centroids)
    x = development[columns]
    y = development[TARGET_COLUMN]
    folds = development["dev_fold"].to_numpy()
    tree_perm = []
    tree_impurity = []
    tree_repeat = []
    forest_perm = []
    forest_impurity = []
    forest_repeat = []
    tree_scores = []
    forest_scores = []
    for fold_id in range(5):
        train = np.flatnonzero(folds != fold_id)
        valid = np.flatnonzero(folds == fold_id)
        train_groups = set(development.iloc[train][GROUP_COLUMN])
        valid_groups = set(development.iloc[valid][GROUP_COLUMN])
        if train_groups & valid_groups:
            raise RuntimeError(f"Fold {fold_id} shares a host star with its training rows.")
        print(f"Fold {fold_id}", flush=True)
        tree = make_tree_pipeline(
            tree_params["max_depth"],
            min_samples_split=int(tree_params["min_samples_split"]),
            min_samples_leaf=int(tree_params["min_samples_leaf"]),
            ccp_alpha=float(tree_params["ccp_alpha"]),
        )
        tree_result = _evaluate_fold(
            tree, x.iloc[train], y.iloc[train], x.iloc[valid], y.iloc[valid], fold_id
        )
        forest = make_forest_pipeline(
            {
                "n_estimators": int(forest_params["n_estimators"]),
                "max_depth": forest_params["max_depth"],
                "max_features": forest_params["max_features"],
                "min_samples_leaf": int(forest_params["min_samples_leaf"]),
                "class_weight": forest_params["class_weight"],
            }
        )
        forest_result = _evaluate_fold(
            forest, x.iloc[train], y.iloc[train], x.iloc[valid], y.iloc[valid], fold_id
        )
        tree_perm.append(tree_result["permutation_mean"])
        tree_impurity.append(tree_result["impurity"])
        tree_repeat.append(tree_result["permutation_std"])
        forest_perm.append(forest_result["permutation_mean"])
        forest_impurity.append(forest_result["impurity"])
        forest_repeat.append(forest_result["permutation_std"])
        tree_scores.append(tree_result["validation_macro_f1"])
        forest_scores.append(forest_result["validation_macro_f1"])
        print(
            f"  tree macro-F1 {tree_result['validation_macro_f1']:.4f}; "
            f"forest macro-F1 {forest_result['validation_macro_f1']:.4f}",
            flush=True,
        )

    saved_tree = pd.read_csv(STAGE3_RESULTS / "comparison_folds.csv")
    saved_tree = saved_tree.loc[saved_tree["model"].eq("selected")].sort_values("fold")
    saved_forest = pd.read_csv(STAGE4_RESULTS / "comparison_folds.csv")
    saved_forest = saved_forest.loc[saved_forest["model"].eq("Tuned forest")].sort_values("fold")
    if not np.allclose(tree_scores, saved_tree["validation_macro_f1"], atol=1e-8):
        raise RuntimeError("Pruned-tree validation macro-F1 does not match Stage 3.")
    if not np.allclose(forest_scores, saved_forest["validation_macro_f1"], atol=1e-8):
        raise RuntimeError("Tuned-forest validation macro-F1 does not match Stage 4.")

    tree_matrix = np.vstack(tree_perm)
    forest_matrix = np.vstack(forest_perm)
    tree_frame = _importance_frame(
        columns, tree_matrix, np.vstack(tree_impurity), np.vstack(tree_repeat)
    )
    forest_frame = _importance_frame(
        columns, forest_matrix, np.vstack(forest_impurity), np.vstack(forest_repeat)
    )
    for frame in (tree_frame, forest_frame):
        frame["centroid"] = frame["column"].isin(centroid_set)

    forest_pairs = _pairwise_spearman(forest_matrix)
    tree_pairs = _pairwise_spearman(tree_matrix)
    forest_ranks = _ranks(forest_matrix)
    tree_ranks = _ranks(tree_matrix)
    forest_order = forest_frame.sort_values("rank")["column"].tolist()
    tree_order = tree_frame.sort_values("rank")["column"].tolist()
    forest_worst = [
        int(forest_ranks[:, columns.index(name)].max()) for name in forest_order[:STABILITY_TOP_N]
    ]
    tree_worst = [
        int(tree_ranks[:, columns.index(name)].max()) for name in tree_order[:STABILITY_TOP_N]
    ]
    forest_stable = ranking_is_stable(float(np.mean(forest_pairs)), forest_worst)
    tree_stable = ranking_is_stable(float(np.mean(tree_pairs)), tree_worst)
    forest_held = sum(rank <= STABILITY_WORST_RANK for rank in forest_worst)
    tree_held = sum(rank <= STABILITY_WORST_RANK for rank in tree_worst)

    forest_share = _positive_share(forest_frame.set_index("column")["mean_importance"], forest_frame.set_index("column")["centroid"])
    tree_share = _positive_share(tree_frame.set_index("column")["mean_importance"], tree_frame.set_index("column")["centroid"])
    forest_top_count = sum(name in centroid_set for name in forest_order[:CENTROID_TOP_N])
    tree_top_count = sum(name in centroid_set for name in tree_order[:CENTROID_TOP_N])
    forest_dominates = centroid_dominates(forest_order, centroid_set, forest_share)
    tree_dominates = centroid_dominates(tree_order, centroid_set, tree_share)

    impurity_spearman = float(
        forest_frame["mean_importance"].corr(forest_frame["mean_impurity"], method="spearman")
    )
    tree_forest_spearman = float(
        forest_frame.set_index("column")
        .loc[columns, "mean_importance"]
        .corr(tree_frame.set_index("column").loc[columns, "mean_importance"], method="spearman")
    )
    shared_top10 = len(set(forest_order[:10]) & set(tree_order[:10]))
    top15 = set(forest_order[:TOP_N])
    pairs = _correlation_pairs(development, columns, top15)
    heatmap_columns = list(forest_order[:TOP_N])
    if not pairs.empty:
        for row in pairs.itertuples(index=False):
            if row.involves_top_feature:
                for name in (row.feature_a, row.feature_b):
                    if name not in heatmap_columns and len(heatmap_columns) < 20:
                        heatmap_columns.append(name)

    def records(frame: pd.DataFrame, limit: int) -> list[dict]:
        return frame.nsmallest(limit, "rank").to_dict(orient="records")

    top3 = forest_frame.nsmallest(3, "rank")
    top3_text = "; ".join(
        f"`{row.column}` stays between fold ranks {int(row.best_fold_rank)} and {int(row.worst_fold_rank)}"
        for row in top3.itertuples(index=False)
    )
    forest_stability_detail = (
        "That correlation uses all 76 features. Many have mean drops near zero, so their order "
        "changes from fold to fold. The first three do not: "
        + top3_text
        + ". From rank 4 downward, a mean rank is not stable when the across-fold standard "
        "deviation is as large as the mean. `figures/person3/feature_importance/forest_rank_by_fold.png` shows "
        "the ten leading mean ranks on each fold."
    )
    definitions = _definition_lines(forest_frame.nsmallest(10, "rank"), dictionary)
    stability = {
        "forest_mean_spearman": float(np.mean(forest_pairs)),
        "forest_min_spearman": float(np.min(forest_pairs)),
        "forest_stable": forest_stable,
        "tree_stable": tree_stable,
        "tree_mean_spearman": float(np.mean(tree_pairs)),
        "tree_min_spearman": float(np.min(tree_pairs)),
        "forest_text": _stability_text("forest", forest_stable, forest_held, forest_worst),
        "tree_text": _stability_text("pruned tree", tree_stable, tree_held, tree_worst),
    }
    centroid_review = {
        "forest_dominates": forest_dominates,
        "tree_dominates": tree_dominates,
        "text": " ".join(
            [
                _centroid_text(
                    "forest",
                    forest_dominates,
                    forest_top_count,
                    forest_order[0],
                    forest_order[0] in centroid_set,
                    forest_share,
                ),
                _centroid_text(
                    "pruned-tree",
                    tree_dominates,
                    tree_top_count,
                    tree_order[0],
                    tree_order[0] in centroid_set,
                    tree_share,
                ),
            ]
        ),
    }
    payload = {
        "n_rows": int(len(development)),
        "n_groups": int(development[GROUP_COLUMN].nunique()),
        "n_features": len(columns),
        "split_sha256": digest,
        "tree_ccp_alpha": float(tree_params["ccp_alpha"]),
        "forest_top": records(forest_frame, TOP_N),
        "tree_top": records(tree_frame, TOP_N),
        "impurity_top": records(
            forest_frame.sort_values("mean_impurity", ascending=False).assign(
                rank=lambda frame: np.arange(1, len(frame) + 1),
                mean_importance=lambda frame: frame["mean_impurity"],
                std_across_folds=lambda frame: frame["std_impurity_across_folds"],
            ),
            TOP_N,
        ),
        "stability": stability,
        "forest_stability_detail": forest_stability_detail,
        "centroid_review": centroid_review,
        "impurity_vs_permutation_spearman": impurity_spearman,
        "tree_forest_spearman": tree_forest_spearman,
        "shared_top10": shared_top10,
        "correlation_pairs": pairs.to_dict(orient="records"),
        "n_strong_pairs": int(len(pairs)),
        "n_strong_pairs_in_top": int(pairs["involves_top_feature"].sum()) if len(pairs) else 0,
        "definitions": definitions,
    }
    # The impurity table reuses the permutation rank fields after renaming. Rebuild ranks
    # from the impurity values so the displayed rank matches the impurity order.
    impurity_view = forest_frame.sort_values("mean_impurity", ascending=False).copy()
    impurity_view["rank"] = np.arange(1, len(impurity_view) + 1)
    impurity_view["mean_importance"] = impurity_view["mean_impurity"]
    impurity_view["std_across_folds"] = impurity_view["std_impurity_across_folds"]
    impurity_order = impurity_view["column"].tolist()
    impurity_worst = []
    impurity_ranks = _ranks(np.vstack(forest_impurity))
    for name in impurity_order[:STABILITY_TOP_N]:
        impurity_worst.append(int(impurity_ranks[:, columns.index(name)].max()))
    impurity_view["best_fold_rank"] = [
        int(impurity_ranks[:, columns.index(name)].min()) for name in impurity_view["column"]
    ]
    impurity_view["worst_fold_rank"] = [
        int(impurity_ranks[:, columns.index(name)].max()) for name in impurity_view["column"]
    ]
    payload["impurity_top"] = impurity_view.head(TOP_N).to_dict(orient="records")

    forest_frame.to_csv(STAGE5_RESULTS / "forest_permutation_importance.csv", index=False)
    tree_frame.to_csv(STAGE5_RESULTS / "tree_permutation_importance.csv", index=False)
    pairs.to_csv(STAGE5_RESULTS / "correlated_pairs.csv", index=False)
    pd.DataFrame(forest_matrix, columns=columns).assign(fold=np.arange(5)).to_csv(
        STAGE5_RESULTS / "forest_fold_permutation.csv", index=False
    )
    pd.DataFrame(tree_matrix, columns=columns).assign(fold=np.arange(5)).to_csv(
        STAGE5_RESULTS / "tree_fold_permutation.csv", index=False
    )
    config = {
        "test_scored": False,
        "models_changed": False,
        "split_sha256": digest,
        "random_seed": RANDOM_SEED,
        "n_repeats": N_REPEATS,
        "scoring": "macro-F1",
        "tree_hyperparameters": tree_params,
        "forest_hyperparameters": forest_params,
        "stability_rule": {
            "minimum_mean_pairwise_spearman": STABILITY_SPEARMAN,
            "top_n": STABILITY_TOP_N,
            "worst_rank_at_most": STABILITY_WORST_RANK,
            "minimum_features_meeting_rank": STABILITY_TOP_REQUIRED,
        },
        "centroid_rule": {
            "top_n": CENTROID_TOP_N,
            "minimum_count": CENTROID_TOP_COUNT,
            "minimum_positive_share": CENTROID_POSITIVE_SHARE,
        },
        "forest_centroid_dominates": forest_dominates,
        "tree_centroid_dominates": tree_dominates,
        "forest_permutation_stable": forest_stable,
        "tree_permutation_stable": tree_stable,
    }
    _json(STAGE5_RESULTS / "model_config.json", config)
    _json(
        STAGE5_RESULTS / "stage5_summary.json",
        {
            "forest_validation_macro_f1": forest_scores,
            "tree_validation_macro_f1": tree_scores,
            "forest_scores": forest_scores,
            "stability": {
                "forest_mean_spearman": stability["forest_mean_spearman"],
                "forest_min_spearman": stability["forest_min_spearman"],
                "forest_stable": forest_stable,
                "tree_mean_spearman": float(np.mean(tree_pairs)),
                "tree_stable": tree_stable,
            },
            "centroid": {
                "forest_top_count": forest_top_count,
                "forest_positive_share": forest_share,
                "forest_dominates": forest_dominates,
                "tree_top_count": tree_top_count,
                "tree_positive_share": tree_share,
                "tree_dominates": tree_dominates,
            },
            "impurity_vs_permutation_spearman": impurity_spearman,
            "tree_forest_spearman": tree_forest_spearman,
            "shared_top10": shared_top10,
            "n_strong_pairs": payload["n_strong_pairs"],
        },
    )
    _plot_ranked(
        forest_frame,
        centroid_set,
        "Tuned forest permutation importance",
        "Mean macro-F1 drop on the validation fold",
        STAGE5_FIGURES / "forest_permutation_importance.png",
    )
    _plot_ranked(
        impurity_view,
        centroid_set,
        "Tuned forest impurity importance",
        "Mean training-fold impurity importance",
        STAGE5_FIGURES / "forest_impurity_importance.png",
    )
    _plot_comparison(
        forest_frame,
        tree_frame,
        centroid_set,
        STAGE5_FIGURES / "tree_vs_forest_permutation.png",
    )
    _plot_rank_paths(
        forest_matrix,
        columns,
        forest_order,
        STAGE5_FIGURES / "forest_rank_by_fold.png",
    )
    _plot_correlation(
        development,
        heatmap_columns,
        STAGE5_FIGURES / "top_feature_correlation.png",
    )
    _write_findings(payload, DOCS / "stage5_findings.md")
    print(stability["forest_text"])
    print(centroid_review["text"])
    print(
        "Stage 5 complete: "
        f"leading forest feature {forest_order[0]}; "
        f"centroid dominates forest={forest_dominates}, tree={tree_dominates}; "
        f"forest stable={forest_stable}."
    )


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
