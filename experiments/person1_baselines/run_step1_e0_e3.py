"""Step 1: E0 to E3 on the saved development folds, both setups, plus nested CV.

E0  Floor: DummyClassifier (most_frequent, stratified).
E1  Logistic Regression: unpenalized, L2, L1, class_weight="balanced".
    C grid 0.001 to 1000. Selected by mean validation macro F1 over the five
    saved development folds. Nested CV (inner folds grouped by host star,
    built from each outer training fold) measures how optimistic that is.
E2  Gaussian Naive Bayes: (a) standardized, (b) signed log, (c) signed log
    plus Spearman pruning decided inside each training fold.
E3  Ablation with the selected LR and NB per setup: 76 features, 56 features
    without the provisional centroid columns, and 76 plus disposition and
    vetting columns (diagnostic only, never real performance).

Setups: "3class" (team setup) and "binary" (CONFIRMED vs FALSE POSITIVE,
CANDIDATE rows dropped, same folds).

Only development rows are fit or scored. Run from the repository root:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python experiments/person1_baselines/run_step1_e0_e3.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from exoplanet.config import ROOT, TARGET_COLUMN
from exoplanet.development import load_development_bundle
from exoplanet.evaluation import cross_validate_estimator
from exoplanet.models.person1_baselines.cv import (
    N_FOLDS,
    nested_inner_scores,
    nested_selection,
    run_cv_many,
    write_experiment,
)
from exoplanet.models.person1_baselines.data import (
    SETUPS,
    add_leakage_columns,
    class_order,
    feature_sets,
    setup_frame,
)
from person1_models import (
    C_GRID,
    DUMMY_KINDS,
    NB_KINDS,
    SEED,
    config_label,
    lr_grid,
    make_model,
    model_config,
)

OUT = ROOT / "results" / "person1"


def key_of(kind, C=None):
    return f"{kind}|{config_label(kind, C)}"


def factory_for(kind, C=None):
    return partial(make_model, kind, C)


def _shared_cv(bundle, columns, kind, C):
    """Official 3-class run through the shared helper."""
    return cross_validate_estimator(replace(bundle, feature_columns=columns), factory_for(kind, C))


def class_distribution(bundle):
    folder = OUT / "class_distribution"
    folder.mkdir(parents=True, exist_ok=True)
    frame = bundle.frame
    by_fold = pd.crosstab(frame["dev_fold"], frame[TARGET_COLUMN])
    by_fold.loc["all"] = by_fold.sum()
    by_fold["total"] = by_fold.sum(axis=1)
    by_fold.to_csv(folder / "development_class_counts_by_fold.csv")
    rows = []
    for setup in SETUPS:
        _, y = setup_frame(frame, setup)
        for c, n in y.value_counts().items():
            rows.append({"setup": setup, "class": c, "count": int(n), "share": n / len(y),
                         "n_rows": len(y)})
    pd.DataFrame(rows).to_csv(folder / "setup_class_counts.csv", index=False)
    return by_fold


def grid_table(results, setup):
    rows = []
    for key, cv in results.items():
        kind, cfg = key.split("|")
        folds = pd.DataFrame(cv.extra_fold_rows)
        row = {"setup": setup, "model": kind, "config": cfg,
               "C": np.nan if cfg in ("-", "C=inf") else float(cfg.split("=")[1]),
               "val_macro_f1_mean": cv.mean(), "val_macro_f1_sd": float(cv.fold_scores().std(ddof=1)),
               "train_macro_f1_mean": cv.mean("train_macro_f1"),
               "macro_f1_gap_mean": cv.mean("train_macro_f1") - cv.mean(),
               "val_accuracy_mean": cv.mean("validation_accuracy"),
               "convergence_warnings_total": int(folds["convergence_warnings"].sum()),
               "n_features_used_mean": float(folds["n_features_used"].mean())}
        for col in ("roc_auc", "roc_auc_ovr_macro", "pr_auc", "brier", "log_loss"):
            if col in folds and folds[col].notna().any():
                row[f"val_{col}_mean"] = float(folds[col].mean())
        if "nonzero_features" in folds:
            row["nonzero_features_mean"] = float(folds["nonzero_features"].mean())
            row["nonzero_features_sd"] = float(folds["nonzero_features"].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def coefficient_table(cv, columns, label):
    """Standardized coefficients, mean and SD over the 5 fold fits."""
    rows = []
    for info in cv.fold_info:
        coef = np.asarray(info["coef"])
        classes = info["model_classes"]
        if coef.shape[0] == 1:  # binary: one row, positive class = classes[1]
            sign = 1.0 if classes[1] == "CONFIRMED" else -1.0
            rows.append(pd.DataFrame({"fold": info["fold"], "target": "CONFIRMED",
                                      "feature": columns, "coef": sign * coef[0]}))
        else:
            for i, c in enumerate(classes):
                rows.append(pd.DataFrame({"fold": info["fold"], "target": c,
                                          "feature": columns, "coef": coef[i]}))
    long = pd.concat(rows)
    table = long.groupby(["target", "feature"])["coef"].agg(
        coef_mean="mean", coef_sd=lambda s: s.std(ddof=1),
        n_folds_nonzero=lambda s: int((s.abs() > 1e-8).sum())).reset_index()
    table.insert(0, "model", label)
    table["abs_mean"] = table["coef_mean"].abs()
    return table.sort_values(["target", "abs_mean"], ascending=[True, False]).drop(columns="abs_mean")


def main(n_jobs: int) -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_development_bundle()
    fsets = feature_sets(bundle)
    leak_frame = add_leakage_columns(bundle.frame)
    print(f"development rows {bundle.n_rows}, host stars {bundle.n_groups}, "
          f"features {len(bundle.feature_columns)}, proposal status {bundle.proposal_status}")
    print(class_distribution(bundle).to_string())

    all_rows, nested_rows, sparsity_rows = [], [], []
    coef_tables, grid_tables, written = [], [], {}
    shared_jobs = []  # 3-class experiments to confirm against the shared helper
    selection = {}

    for setup in SETUPS:
        classes = class_order(setup)
        frame, y = setup_frame(bundle.frame, setup)
        F = fsets["F76"]
        print(f"\n== {setup}: rows {len(frame)}, classes {y.value_counts().to_dict()}")

        configs = [(k, None) for k in DUMMY_KINDS] + lr_grid() + [(k, None) for k in NB_KINDS]
        jobs = {
            key_of(k, C): dict(frame=frame, y=y, columns=F, factory=factory_for(k, C),
                               setup=setup, classes=classes,
                               log_loss_applicable=k not in DUMMY_KINDS)
            for k, C in configs
        }
        t = time.time()
        res = run_cv_many(jobs, n_jobs=n_jobs)
        print(f"  F76 grid: {len(jobs)} configurations x 5 folds in {time.time() - t:.0f} s")

        grid = grid_table(res, setup)
        grid_tables.append(grid)
        for _, r in grid.iterrows():
            print(f"  {r['model']:20s} {r['config']:8s} val macroF1={r['val_macro_f1_mean']:.4f} "
                  f"conv_warn={r['convergence_warnings_total']}"
                  + (f" nonzero={r['nonzero_features_mean']:.1f}"
                     if pd.notna(r.get("nonzero_features_mean", np.nan)) else ""))

        # ---------------- selection on the five folds (non-nested)
        def best_of(candidates):
            return max(candidates, key=lambda kc: res[key_of(*kc)].mean())

        sel_l2 = best_of([("lr_l2", C) for C in C_GRID])
        sel_l1 = best_of([("lr_l1", C) for C in C_GRID])
        lr_final = [("lr_none", None), sel_l2, sel_l1]
        best_lr = best_of(lr_final)
        best_nb = best_of([(k, None) for k in NB_KINDS])
        selection[setup] = {"lr_l2": sel_l2, "lr_l1": sel_l1, "best_lr": best_lr, "best_nb": best_nb}
        print(f"  selected L2 {sel_l2}, L1 {sel_l1}, best LR {best_lr}, best NB {best_nb}")

        for col in ("lr_l1",):
            for C in C_GRID:
                cv = res[key_of(col, C)]
                folds = pd.DataFrame(cv.extra_fold_rows)
                sparsity_rows.append({"setup": setup, "C": C,
                                      "nonzero_features_mean": folds["nonzero_features"].mean(),
                                      "nonzero_features_sd": folds["nonzero_features"].std(ddof=1),
                                      "nonzero_min": int(folds["nonzero_features"].min()),
                                      "nonzero_max": int(folds["nonzero_features"].max()),
                                      "val_macro_f1_mean": cv.mean(),
                                      "convergence_warnings_total": int(folds["convergence_warnings"].sum())})

        # ---------------- write E0, E1, E2
        to_write = [("E0", k, None) for k in DUMMY_KINDS] + [("E1", k, C) for k, C in lr_final] \
            + [("E2", k, None) for k in NB_KINDS]
        for exp, k, C in to_write:
            name = f"{exp.lower()}_{k}_{setup}"
            cfg = model_config(k, C)
            cfg.update({"experiment": exp, "feature_set": "F76"})
            if exp == "E1" and k != "lr_none":
                cfg["selection"] = (f"C chosen from {list(C_GRID)} by mean validation macro F1 over "
                                    "the 5 saved development folds; nested estimate in nested_cv/")
            written[(setup, exp, "F76", k)] = (name, res[key_of(k, C)], cfg, k, C, F)

        # ---------------- E3 ablation with the selected LR and NB
        e3_jobs, e3_meta = {}, {}
        for fam, (k, C) in (("lr", best_lr), ("nb", best_nb)):
            for fset in ("F56", "F76_plus_leakage"):
                src = leak_frame if fset == "F76_plus_leakage" else bundle.frame
                f_set, y_set = setup_frame(src, setup)
                key = f"{fam}|{fset}"
                e3_jobs[key] = dict(frame=f_set, y=y_set, columns=fsets[fset],
                                    factory=factory_for(k, C), setup=setup, classes=classes)
                e3_meta[key] = (fam, fset, k, C)
        e3 = run_cv_many(e3_jobs, n_jobs=n_jobs)
        for key, cv in e3.items():
            fam, fset, k, C = e3_meta[key]
            tag = "f56_no_centroid" if fset == "F56" else "leakage_diagnostic"
            name = f"e3_{tag}_{fam}_{setup}"
            cfg = model_config(k, C)
            cfg.update({"experiment": "E3", "feature_set": fset,
                        "configuration_source": "selected on F76 in E1/E2, not re-tuned"})
            if fset == "F76_plus_leakage":
                cfg["diagnostic_only"] = True
                cfg["note"] = ("Adds disposition and vetting outputs that encode the label. "
                               "Diagnostic only, never real performance.")
            written[(setup, "E3", fset, k)] = (name, cv, cfg, k, C, fsets[fset])
            print(f"  E3 {fam} {fset:18s} val macroF1={cv.mean():.4f}")

        # ---------------- E3 ablation table, F76 reference reused from E1/E2
        for fam, (k, C) in (("lr", best_lr), ("nb", best_nb)):
            ref = res[key_of(k, C)]
            for fset, cv in (("F76", ref), ("F56", e3[f"{fam}|F56"]),
                             ("F76_plus_leakage", e3[f"{fam}|F76_plus_leakage"])):
                all_rows.append({"table": "e3_ablation", "setup": setup, "model": k,
                                 "config": config_label(k, C), "feature_set": fset,
                                 "val_macro_f1_mean": cv.mean(),
                                 "val_macro_f1_sd": float(cv.fold_scores().std(ddof=1)),
                                 "diagnostic_only": fset == "F76_plus_leakage"})

        # Paired fold-by-fold comparisons are built from the saved fold metrics
        # by run_paired_comparisons.py.

        # ---------------- coefficients
        for label, (k, C) in (("lr_none", ("lr_none", None)), (f"lr_l2 {config_label(*sel_l2)}", sel_l2),
                              (f"lr_l1 {config_label(*sel_l1)}", sel_l1)):
            table = coefficient_table(res[key_of(k, C)], F, label)
            table.insert(0, "setup", setup)
            coef_tables.append(table)

        # ---------------- nested CV over the same candidates
        t = time.time()
        candidate_cfgs = lr_grid() + [(k, None) for k in NB_KINDS]
        factories = {key_of(k, C): factory_for(k, C) for k, C in candidate_cfgs}
        inner = nested_inner_scores(frame, y, F, factories, classes, seed=SEED, n_jobs=n_jobs)
        inner.insert(0, "setup", setup)
        nested_dir = OUT / "nested_cv"
        nested_dir.mkdir(parents=True, exist_ok=True)
        inner.to_csv(nested_dir / f"inner_scores_{setup}.csv", index=False)
        outer_scores = {key_of(k, C): res[key_of(k, C)].fold_scores() for k, C in candidate_cfgs}
        families = {
            "LR L2: choose C": [key_of("lr_l2", C) for C in C_GRID],
            "LR L1: choose C": [key_of("lr_l1", C) for C in C_GRID],
            "LR: choose penalty and C": [key_of(k, C) for k, C in lr_grid()],
            "NB: choose variant": [key_of(k) for k in NB_KINDS],
            "LR or NB: choose everything": list(factories),
        }
        for fam, cands in families.items():
            sel = nested_selection(inner, outer_scores, cands)
            nested_rows.append({"setup": setup, "selection": fam, **{
                k: (json.dumps(v) if isinstance(v, list) else v) for k, v in sel.items()
                if k != "candidates"}, "n_candidates": len(cands)})
            print(f"  nested {fam:30s} nested={sel['nested_mean']:.4f} "
                  f"non-nested={sel['non_nested_mean']:.4f} optimism={sel['optimism']:+.4f} "
                  f"chosen={sel['chosen_per_outer_fold']}")
        print(f"  nested CV inner fits in {time.time() - t:.0f} s")

    # ---------------- confirm 3-class runs against the shared helper, then write
    shared_keys = [key for key in written if key[0] == "3class" and key[2] != "F76_plus_leakage"]
    shared = Parallel(n_jobs=n_jobs)(
        delayed(_shared_cv)(bundle, written[key][5], written[key][3], written[key][4])
        for key in shared_keys
    )
    shared_by_key = dict(zip(shared_keys, shared))
    summary_rows = []
    for key, (name, cv, cfg, k, C, cols) in written.items():
        summary = None
        if key in shared_by_key:
            summary = shared_by_key[key]
            mine = cv.oof["y_pred"]
            theirs = summary["oof_prediction"].astype(str)
            if not mine.equals(theirs.loc[mine.index]):
                raise RuntimeError(f"{name}: OOF labels differ from the shared helper.")
            if abs(summary["mean_validation_macro_f1"] - cv.mean()) > 1e-12:
                raise RuntimeError(f"{name}: macro F1 differs from the shared helper.")
            cfg["summary_source"] = "exoplanet.evaluation.cross_validate_estimator"
        else:
            cfg["summary_source"] = ("person1 fold loop on the saved development folds "
                                     "(binary setup or leakage diagnostic)")
        path = write_experiment(name, cv, cfg, summary=summary)
        extra = json.loads((OUT / name / "extra_metrics.json").read_text())
        fm = extra["fold_mean_sd"]
        row = {"experiment": name, "setup": key[0], "stage": key[1], "feature_set": key[2],
               "model": k, "config": config_label(k, C),
               "diagnostic_only": key[2] == "F76_plus_leakage", "path": f"results/person1/{name}"}
        for metric in ("macro_f1", "accuracy", "balanced_accuracy", "weighted_f1", "mcc",
                       "cohen_kappa", "train_macro_f1", "macro_f1_gap", "roc_auc",
                       "roc_auc_ovr_macro", "pr_auc", "pr_auc_ovr_macro", "brier",
                       "brier_multiclass", "log_loss", "ece_10bin"):
            if metric in fm:
                row[f"{metric}_mean"] = fm[metric]["mean"]
                row[f"{metric}_sd"] = fm[metric]["std"]
        for metric in ("macro_f1", "accuracy", "roc_auc", "roc_auc_ovr_macro", "pr_auc",
                       "brier", "brier_multiclass", "log_loss"):
            if metric in extra["pooled_oof"]:
                row[f"{metric}_pooled"] = extra["pooled_oof"][metric]
        for c in cv.classes:
            row[f"recall_{c.replace(' ', '_')}_pooled"] = extra["pooled_oof"][f"recall_{c.replace(' ', '_')}"]
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(OUT / "step1_summary.csv", index=False)
    pd.concat(grid_tables).to_csv(OUT / "e1_e2_grid.csv", index=False)
    pd.DataFrame(sparsity_rows).to_csv(OUT / "e1_l1_sparsity.csv", index=False)
    pd.concat(coef_tables).to_csv(OUT / "e1_lr_coefficients.csv", index=False)
    pd.DataFrame(all_rows).to_csv(OUT / "e3_ablation.csv", index=False)
    pd.DataFrame(nested_rows).to_csv(OUT / "nested_cv" / "nested_vs_non_nested.csv", index=False)
    (OUT / "step1_selection.json").write_text(json.dumps(
        {s: {k: {"model": v[0], "config": config_label(*v)} for k, v in d.items()}
         for s, d in selection.items()}, indent=2))
    print(f"\nWrote {len(summary_rows)} experiment folders. Runtime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=-1)
    main(parser.parse_args().n_jobs)
