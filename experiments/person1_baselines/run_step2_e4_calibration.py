"""Step 2: E4 probability calibration in the binary vetting setup.

Models: the selected binary logistic regression and the two log-transformed
Naive Bayes variants (b) and (c). Each is recalibrated with Platt scaling
(sigmoid) and with isotonic regression through CalibratedClassifierCV, using
four inner folds grouped by host star and built from the outer training fold
only. The uncalibrated results are read from the Step 1 cache, not refit.

Reported before vs after: Brier score, log loss, 10-bin expected calibration
error, ROC-AUC, PR-AUC, and macro F1 at the default 0.5 threshold, as means
over the five development folds, pooled out of fold, and as paired
fold-by-fold differences. Reliability tables (10 equal-width bins, pooled
out of fold) are saved for the reliability diagrams.

Run from the repository root after Step 1:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python experiments/person1_baselines/run_step2_e4_calibration.py
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial

import numpy as np
import pandas as pd

from exoplanet.config import ROOT
from exoplanet.development import load_development_bundle
from exoplanet.models.person1_baselines.calibration import (
    N_CALIBRATION_FOLDS,
    make_calibrated,
    reliability_bins,
)
from exoplanet.models.person1_baselines.cv import run_cv_many, write_experiment
from exoplanet.models.person1_baselines.data import POSITIVE, class_order, setup_frame
from exoplanet.models.person1_baselines.metrics import corrected_paired_t
from person1_models import config_label, make_model, model_config

OUT = ROOT / "results" / "person1"
METHODS = ("sigmoid", "isotonic")
COMPARE = ("brier", "log_loss", "ece_10bin", "roc_auc", "pr_auc", "macro_f1", "accuracy")


def step1_folder(kind: str, best_lr: tuple) -> str:
    if kind.startswith("lr_"):
        return f"e1_{best_lr[0]}_binary"
    return f"e2_{kind}_binary"


def main(n_jobs: int) -> None:
    t0 = time.time()
    selection = json.loads((OUT / "step1_selection.json").read_text())["binary"]
    lr = selection["best_lr"]
    lr_kind = lr["model"]
    lr_C = None if lr["config"] in ("C=inf", "-") else float(lr["config"].split("=")[1])
    models = [(lr_kind, lr_C), ("nb_b_signed_log", None), ("nb_c_log_spearman", None)]

    bundle = load_development_bundle()
    frame, y = setup_frame(bundle.frame, "binary")
    classes = class_order("binary")
    F = list(bundle.feature_columns)

    jobs = {}
    for kind, C in models:
        for method in METHODS:
            jobs[f"{kind}|{method}"] = dict(
                frame=frame, y=y, columns=F,
                factory=partial(make_calibrated, partial(make_model, kind, C), method),
                setup="binary", classes=classes,
            )
    results = run_cv_many(jobs, n_jobs=n_jobs)
    print(f"calibrated runs: {len(jobs)} x 5 folds in {time.time() - t0:.0f} s")

    folder = OUT / "e4_calibration"
    folder.mkdir(parents=True, exist_ok=True)
    rel_rows, cmp_rows, paired_rows, fold_rows = [], [], [], []
    y_bin = None
    for kind, C in models:
        base_name = step1_folder(kind, (lr_kind, lr_C))
        base_oof = pd.read_csv(OUT / base_name / "oof_predictions.csv.gz")
        base_folds = pd.read_csv(OUT / base_name / "extra_fold_metrics.csv")
        base_pooled = json.loads((OUT / base_name / "extra_metrics.json").read_text())["pooled_oof"]
        variants = [("uncalibrated", base_oof, base_folds, base_pooled, base_name)]
        for method in METHODS:
            cv = results[f"{kind}|{method}"]
            name = f"e4_{kind}_{method}_binary"
            cfg = model_config(kind, C)
            cfg.update({
                "experiment": "E4", "feature_set": "F76",
                "calibration": {
                    "method": method, "wrapper": "sklearn CalibratedClassifierCV",
                    "ensemble": True, "inner_folds": N_CALIBRATION_FOLDS,
                    "inner_split": "StratifiedGroupKFold on kepid, shuffle, seed 245, built "
                                   "from the outer training fold only",
                },
                "uncalibrated_reference": f"results/person1/{base_name}",
            })
            write_experiment(name, cv, cfg)
            extra = json.loads((OUT / name / "extra_metrics.json").read_text())
            variants.append((method, cv.oof.reset_index(drop=True), pd.DataFrame(cv.extra_fold_rows),
                             extra["pooled_oof"], name))

        label = f"{kind} {config_label(kind, C)}"
        base_f = variants[0][2].set_index("fold")
        for vname, oof, folds, pooled, path in variants:
            y_true = oof["y_true"].eq(POSITIVE).astype(int)
            p = oof["p_CONFIRMED"].to_numpy()
            table = reliability_bins(y_true, p)
            table.insert(0, "variant", vname)
            table.insert(0, "model", label)
            rel_rows.append(table)
            f = folds.set_index("fold")
            row = {"model": label, "variant": vname, "path": f"results/person1/{path}"}
            for m in COMPARE:
                row[f"{m}_mean"] = float(f[m].mean())
                row[f"{m}_sd"] = float(f[m].std(ddof=1))
                row[f"{m}_pooled"] = pooled[m]
            row["mean_p_confirmed"] = float(p.mean())
            row["observed_confirmed_share"] = float(y_true.mean())
            cmp_rows.append(row)
            for k, r in f.iterrows():
                fold_rows.append({"model": label, "variant": vname, "fold": k,
                                  **{m: r[m] for m in COMPARE}})
            if vname != "uncalibrated":
                n_val = 5876 / 5
                n_tr = 5876 - n_val
                for m in COMPARE:
                    d = (f[m] - base_f[m]).to_numpy()
                    paired_rows.append({"model": label, "comparison": f"{vname} minus uncalibrated",
                                        "metric": m, **{f"fold{k}_diff": float(d[k]) for k in range(5)},
                                        **corrected_paired_t(d, n_tr, n_val, m)})

    comparison = pd.DataFrame(cmp_rows)
    comparison.to_csv(folder / "before_after.csv", index=False)
    pd.concat(rel_rows).to_csv(folder / "reliability_bins.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(folder / "paired_differences.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(folder / "fold_metrics_all_variants.csv", index=False)

    # Model for the ranked CANDIDATE list: lowest pooled Brier among calibrated variants.
    calibrated = comparison.loc[comparison["variant"].ne("uncalibrated")]
    best = calibrated.sort_values(["brier_pooled", "log_loss_pooled"]).iloc[0]
    kind = best["model"].split(" ")[0]
    choice = {
        "model": kind,
        "config": best["model"].split(" ", 1)[1],
        "C": lr_C if kind == lr_kind else None,
        "method": best["variant"],
        "criterion": "lowest pooled out-of-fold Brier score among calibrated binary variants "
                     "(log loss as tie-break)",
        "brier_pooled": best["brier_pooled"],
        "log_loss_pooled": best["log_loss_pooled"],
        "pr_auc_pooled": best["pr_auc_pooled"],
        "path": best["path"],
    }
    (folder / "selected_calibrated_model.json").write_text(json.dumps(choice, indent=2))
    pd.set_option("display.width", 200)
    show = ["model", "variant", "brier_mean", "log_loss_mean", "ece_10bin_mean", "roc_auc_mean",
            "pr_auc_mean", "macro_f1_mean", "mean_p_confirmed"]
    print(comparison[show].round(4).to_string(index=False))
    print("selected for the ranked list:", choice)
    print(f"runtime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=-1)
    main(parser.parse_args().n_jobs)
