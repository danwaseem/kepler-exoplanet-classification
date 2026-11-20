"""Report figures for Person 1, drawn only from saved results (no refitting).

Writes PNG files at 200 dpi to figures/person1/. Widths follow an IEEE
two-column page: 3.5 in for one column, 7.16 in for both.

Run from the repository root after Steps 1 to 4:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python experiments/person1_baselines/make_figures.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from exoplanet.config import ROOT

RES = ROOT / "results" / "person1"
FIG = ROOT / "figures" / "person1"
ONE_COL, TWO_COL = 3.5, 7.16

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, DARK_GRAY, INK, MUTED = "#a3a29a", "#6b6a63", "#1f1f1d", "#5f5e58"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CLASS_COLOR = {"CONFIRMED": BLUE, "CANDIDATE": AQUA, "FALSE POSITIVE": ORANGE}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f7f9fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "font.family": "DejaVu Sans",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "figure.dpi": 100,
})


def save(fig, name):
    """Save at 200 dpi, then store as a 256-colour palette PNG to keep files small."""
    from PIL import Image

    path = FIG / name
    fig.savefig(path)
    plt.close(fig)
    with Image.open(path) as im:
        small = im.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    small.save(path, optimize=True)
    print(f"  {name}: {path.stat().st_size / 1024:.0f} KB")


def short(model):
    return {
        "dummy_most_frequent": "Majority", "dummy_stratified": "Stratified random",
        "lr_none": "LR, no penalty", "lr_l2": "LR, L2", "lr_l1": "LR, L1",
        "nb_a_standardized": "NB (a) standardized", "nb_b_signed_log": "NB (b) signed log",
        "nb_c_log_spearman": "NB (c) log + pruning",
    }.get(model, model)


def setup_title(setup):
    return "Three classes" if setup == "3class" else "Binary: CONFIRMED vs FALSE POSITIVE"


# ---------------------------------------------------------------- figures
def class_distribution():
    counts = pd.read_csv(RES / "class_distribution" / "development_class_counts_by_fold.csv",
                         index_col=0)
    allrow = counts.loc["all"]
    classes = ["FALSE POSITIVE", "CONFIRMED", "CANDIDATE"]
    vals = [int(allrow[c]) for c in classes]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(ONE_COL, 1.45))
    y = np.arange(len(classes))[::-1]
    ax.barh(y, vals, color=[CLASS_COLOR[c] for c in classes], height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + total * 0.01, yi, f"{v:,}  ({v / total:.1%})", va="center", fontsize=7.5,
                color=INK)
    ax.set_yticks(y, classes)
    ax.set_xlim(0, max(vals) * 1.32)
    ax.set_xlabel("Development rows with measured transit depth")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(f"Class balance, {total:,} development KOIs", loc="left")
    save(fig, "class_distribution.png")


def _cm_panel(ax, cm, labels, title):
    cm = np.asarray(cm, dtype=float)
    share = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{int(v):,}\n{share[r, c]:.0%}", ha="center", va="center", fontsize=7,
                color="white" if share[r, c] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def confusion_matrices(sel):
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL * 0.78, 5.0))
    abbrev3 = ["CONF", "CAND", "FP"]
    for i, setup in enumerate(("3class", "binary")):
        for j, fam in enumerate(("best_lr", "best_nb")):
            model = sel[setup][fam]["model"]
            stage = "e1" if fam == "best_lr" else "e2"
            s = json.loads((RES / f"{stage}_{model}_{setup}" / "summary.json").read_text())
            if setup == "3class":
                cm, labels = s["confusion_matrix"], abbrev3
            else:
                cm, labels = s["binary_confusion_matrix"]["matrix"], ["CONF", "FP"]
            cfg = sel[setup][fam]["config"]
            name = short(model) + ("" if cfg in ("-", "C=inf") else f" {cfg}")
            _cm_panel(axes[i, j], cm, labels,
                      f"{'3-class' if setup == '3class' else 'Binary'}: {name}\n"
                      f"macro F1 {s['mean_validation_macro_f1']:.3f}")
    fig.text(0.0, -0.01, "Pooled out-of-fold predictions over the 5 development folds. "
             "Cell shading and percentages are row shares (recall per true class).",
             fontsize=7, color=MUTED)
    fig.tight_layout(h_pad=1.5)
    save(fig, "confusion_matrices.png")


def model_comparison(sel):
    summ = pd.read_csv(RES / "step1_summary.csv")
    summ = summ.loc[summ["feature_set"].eq("F76")]
    order = ["dummy_most_frequent", "dummy_stratified", "nb_a_standardized", "nb_b_signed_log",
             "nb_c_log_spearman", "lr_l1", "lr_l2", "lr_none"]
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.45), sharey=True)
    y = np.arange(len(order))
    for ax, setup in zip(axes, ("3class", "binary")):
        sub = summ.loc[summ["setup"].eq(setup)].set_index("model").loc[order]
        for yi, (model, r) in zip(y, sub.iterrows()):
            folds = pd.read_csv(RES / r["experiment"] / "fold_metrics.csv")["validation_macro_f1"]
            color = GRAY if model.startswith("dummy") else (BLUE if model.startswith("lr") else ORANGE)
            ax.scatter(folds, np.full(len(folds), yi), s=9, color=color, alpha=0.45, lw=0,
                       zorder=2)
            ax.errorbar(r["macro_f1_mean"], yi, xerr=r["macro_f1_sd"], fmt="o", ms=5,
                        color=color, mec="white", mew=0.8, elinewidth=1.2, capsize=0, zorder=3)
            # The selected C differs by setup, so it is printed beside the value.
            cfg = "" if r["config"] in ("-", "C=inf") else f"  ({r['config']})"
            ax.text(r["macro_f1_mean"], yi + 0.33, f"{r['macro_f1_mean']:.3f}{cfg}",
                    ha="center", fontsize=6.3, color=MUTED)
        ax.set_xlim(0.15, 1.02)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel("Validation macro F1")
        ax.set_title(setup_title(setup), loc="left")
    axes[0].set_yticks(y, [short(m) + (", tuned C" if m in ("lr_l1", "lr_l2") else "")
                           for m in order])
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec="white", ms=5, label=l)
               for c, l in ((BLUE, "logistic regression"), (ORANGE, "Gaussian Naive Bayes"),
                            (GRAY, "dummy floor"))]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=DARK_GRAY, alpha=0.45, ms=3,
                              label="single fold (mean ± 1 SD shown large)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06),
               fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save(fig, "lr_vs_nb_comparison.png")


def centroid_ablation():
    ab = pd.read_csv(RES / "e3_ablation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.1), sharey=False)
    sets = [("F56", "56 features\n(no centroid)"), ("F76", "76 features\n(primary)"),
            ("F76_plus_leakage", "76 + vetting outputs\n(diagnostic only)")]
    for ax, setup in zip(axes, ("3class", "binary")):
        sub = ab.loc[ab["setup"].eq(setup)]
        width = 0.36
        for j, (fam, color) in enumerate((("lr", BLUE), ("nb", ORANGE))):
            rows = sub.loc[sub["model"].str.startswith(fam)]
            model = rows["model"].iloc[0]
            x = np.arange(3) + (j - 0.5) * width
            vals = [rows.loc[rows["feature_set"].eq(f), "val_macro_f1_mean"].iloc[0] for f, _ in sets]
            sds = [rows.loc[rows["feature_set"].eq(f), "val_macro_f1_sd"].iloc[0] for f, _ in sets]
            bars = ax.bar(x, vals, width * 0.92, color=color, yerr=sds,
                          error_kw={"elinewidth": 0.8, "ecolor": DARK_GRAY},
                          label=short(model) + ("" if rows["config"].iloc[0] in ("-", "C=inf")
                                                else f" ({rows['config'].iloc[0]})"))
            bars[2].set_hatch("////")
            bars[2].set_edgecolor("white")
            bars[2].set_alpha(0.55)
            for xi, v in zip(x, vals):
                ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.5, color=INK)
        ax.set_xticks(range(3), [s for _, s in sets])
        ax.set_ylim(0.5 if setup == "3class" else 0.75, 1.04)
        ax.set_ylabel("Validation macro F1")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(setup_title(setup), loc="left", pad=4)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08),
               fontsize=7)
    fig.tight_layout(w_pad=1.5)
    save(fig, "centroid_ablation.png")


def coefficients(sel):
    co = pd.read_csv(RES / "e1_lr_coefficients.csv")
    l2 = sel["binary"]["lr_l2"]
    label = f"lr_l2 {l2['config']}"
    sub = co.loc[co["setup"].eq("binary") & co["model"].eq(label)].copy()
    sub["abs"] = sub["coef_mean"].abs()
    sub = sub.nlargest(15, "abs").iloc[::-1]
    prov = set(pd.read_csv(ROOT / "data" / "processed" / "proposed_leakage_aware_features.csv")
               .query("provisional")["column"])
    fig, ax = plt.subplots(figsize=(ONE_COL, 3.3))
    y = np.arange(len(sub))
    colors = [BLUE if v > 0 else ORANGE for v in sub["coef_mean"]]
    ax.barh(y, sub["coef_mean"], xerr=sub["coef_sd"], color=colors, height=0.66,
            error_kw={"elinewidth": 0.8, "ecolor": DARK_GRAY})
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(y, [f + (" *" if f in prov else "") for f in sub["feature"]], fontsize=7)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Standardized coefficient (mean ± SD over 5 folds)")
    ax.set_title(f"Binary LR, L2 {l2['config']}: 15 largest coefficients", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="toward CONFIRMED"),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="toward FALSE POSITIVE"),
               plt.Line2D([], [], ls="", label="* provisional centroid column")]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, handlelength=1.0)
    save(fig, "lr_coefficients_binary.png")


def l1_sparsity():
    sp = pd.read_csv(RES / "e1_l1_sparsity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL * 0.8, 1.9))
    for setup, color, marker in (("3class", BLUE, "o"), ("binary", ORANGE, "s")):
        s = sp.loc[sp["setup"].eq(setup)]
        lab = "3-class" if setup == "3class" else "binary"
        axes[0].plot(s["C"], s["nonzero_features_mean"], marker=marker, ms=4, lw=1.6,
                     color=color, label=lab)
        axes[1].plot(s["C"], s["val_macro_f1_mean"], marker=marker, ms=4, lw=1.6,
                     color=color, label=lab)
    axes[0].set_ylabel("Features with nonzero weight")
    axes[0].set_ylim(0, 80)
    axes[1].set_ylabel("Validation macro F1")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("C (inverse regularization strength)")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower right")
    axes[0].set_title("L1 logistic regression: sparsity", loc="left")
    axes[1].set_title("L1 logistic regression: accuracy cost", loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, "l1_sparsity.png")


def reliability():
    rb = pd.read_csv(RES / "e4_calibration" / "reliability_bins.csv")
    ba = pd.read_csv(RES / "e4_calibration" / "before_after.csv")
    models = list(dict.fromkeys(rb["model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(TWO_COL, 2.75), sharey=True)
    styles = {"uncalibrated": (DARK_GRAY, "o"), "sigmoid": (BLUE, "s"), "isotonic": (ORANGE, "D")}
    names = {"uncalibrated": "uncalibrated", "sigmoid": "Platt (sigmoid)", "isotonic": "isotonic"}
    for ax, model in zip(axes, models):
        ax.plot([0, 1], [0, 1], color=AXIS, lw=0.9, ls="--", zorder=1)
        stats = []
        for variant, (color, marker) in styles.items():
            s = rb.loc[rb["model"].eq(model) & rb["variant"].eq(variant) & rb["count"].gt(0)]
            b = ba.loc[ba["model"].eq(model) & ba["variant"].eq(variant)].iloc[0]
            ax.plot(s["mean_predicted"], s["observed_fraction"], marker=marker, ms=3.5, lw=1.5,
                    color=color, zorder=3, label=names[variant])
            stats.append(f"{b['brier_pooled']:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_axisbelow(True)
        name = model.split(" ")[0]
        ax.set_title(short(name) + ("" if model.endswith(("C=inf", "-")) else " " + model.split(" ", 1)[1]),
                     loc="left")
        ax.set_xlabel("Mean predicted P(CONFIRMED)\nBrier: " + " / ".join(stats), fontsize=7.5)
    axes[0].set_ylabel("Observed share CONFIRMED")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05),
               fontsize=7.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, "reliability_diagrams.png")


def model_tick(name: str) -> str:
    """'LR lr_l2 C=1000' -> 'LR, L2\\nC=1000'; calibrated names get a third line."""
    calibrated = " + " in name
    base = name.split(" + ")[0].replace("LR ", "")
    if base.startswith("NB"):
        label = "NB (b)\nsigned log"
    else:
        kind, _, cfg = base.partition(" ")
        label = short(kind) + ("" if cfg in ("", "-", "C=inf") else f"\n{cfg}")
    if calibrated:
        label += "\n+ " + name.split(" + ")[1].replace(" calibration", "")
    return label


def temporal():
    tm = pd.read_csv(RES / "e5_temporal" / "temporal_metrics.csv")
    ref = pd.read_csv(RES / "e5_temporal" / "reference_metrics.csv")
    main = tm.loc[tm["training_set"].eq("all old labels")]
    models = [m for m in main["model"] if not m.startswith(("always", "constant"))]
    always = main.loc[main["model"].str.startswith("always")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.5))
    contexts = [
        ("current table, saved dev folds", ref.loc[ref["evaluation"].str.startswith("current")], GRAY),
        ("old table, grouped CV", ref.loc[ref["evaluation"].str.startswith("grouped")], DARK_GRAY),
        ("forward in time: old candidates", main, BLUE),
    ]
    for ax, metric, label in ((axes[0], "macro_f1", "Macro F1"), (axes[1], "roc_auc", "ROC-AUC")):
        width = 0.26
        for j, (cname, frame, color) in enumerate(contexts):
            vals = [frame.loc[frame["model"].eq(m), metric].iloc[0] for m in models]
            x = np.arange(len(models)) + (j - 1) * width
            ax.bar(x, vals, width * 0.92, color=color, label=cname)
            for xi, v in zip(x, vals):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=5.6, color=INK)
        if metric == "macro_f1":
            ax.axhline(always["macro_f1"], color=ORANGE, lw=1.2, ls="--")
            ax.text(len(models) - 0.5, always["macro_f1"] + 0.015,
                    f"always planet {always['macro_f1']:.2f}", ha="right", fontsize=6.5, color=INK)
        else:
            ax.axhline(0.5, color=ORANGE, lw=1.2, ls="--")
            ax.text(len(models) - 0.5, 0.515, "chance 0.50", ha="right", fontsize=6.5, color=INK)
        ax.set_xticks(range(len(models)), [model_tick(m) for m in models], fontsize=6.5)
        ax.set_ylim(0.3, 1.05)
        ax.set_ylabel(label)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06),
               fontsize=7)
    fig.tight_layout(w_pad=2.0)
    save(fig, "e5_temporal_validation.png")


def crosstab():
    t = pd.read_csv(RES / "label_crosstab" / "crosstab_all_kois.csv", index_col=0)
    labels = ["CONF", "CAND", "FP"]
    fig, ax = plt.subplots(figsize=(ONE_COL * 0.82, 2.4))
    M = t.to_numpy(dtype=float)
    share = M / M.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for (r, c), v in np.ndenumerate(M):
        ax.text(c, r, f"{int(v):,}", ha="center", va="center", fontsize=7.5,
                color="white" if share[r, c] > 0.55 else INK)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Current label (NASA 2026-10-06)")
    ax.set_ylabel("Old label (Kaggle copy)")
    changed = int(M.sum() - np.trace(M))
    ax.set_title(f"Label changes between versions: {changed:,} of {int(M.sum()):,}", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "label_crosstab.png")


def ranked_candidates():
    rk = pd.read_csv(RES / "ranked_candidates_development.csv")
    fig, ax = plt.subplots(figsize=(ONE_COL, 1.8))
    bins = np.linspace(0, 1, 21)
    ax.hist(rk["p_confirmed_calibrated"], bins=bins, color=BLUE, edgecolor="white", lw=0.8)
    for cut in (0.1, 0.5, 0.9):
        ax.axvline(cut, color=AXIS, lw=0.8, ls="--")
    ax.set_xlabel("Calibrated P(CONFIRMED)")
    ax.set_ylabel("Development CANDIDATEs")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(f"Scores for {len(rk):,} unresolved development CANDIDATEs", loc="left")
    save(fig, "ranked_candidates_histogram.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sel = json.loads((RES / "step1_selection.json").read_text())
    for fn in (class_distribution, lambda: confusion_matrices(sel), lambda: model_comparison(sel),
               centroid_ablation, lambda: coefficients(sel), l1_sparsity,
               reliability, temporal, crosstab, ranked_candidates):
        fn()
    total = sum(p.stat().st_size for p in FIG.glob("*.png"))
    print(f"total figure size {total / 1024:.0f} KB")


if __name__ == "__main__":
    main()
