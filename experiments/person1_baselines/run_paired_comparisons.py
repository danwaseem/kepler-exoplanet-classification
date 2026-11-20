"""Paired fold-by-fold comparisons between Step 1 models, from saved results.

All models share the same five saved development folds, so differences are
paired by fold. For each pair and metric this writes the five per-fold
differences (first minus second), their mean and SD, how many folds favour
the first model, and the Nadeau-Bengio corrected resampled t statistic. With
five folds the p-value is a rough guide only.

Reads results/person1/<experiment>/fold_metrics.csv and extra_fold_metrics.csv;
nothing is refit. Run after Step 1:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python experiments/person1_baselines/run_paired_comparisons.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exoplanet.config import ROOT
from exoplanet.models.person1_baselines.metrics import corrected_paired_t

RES = ROOT / "results" / "person1"


def fold_table(experiment: str) -> pd.DataFrame:
    shared = pd.read_csv(RES / experiment / "fold_metrics.csv").set_index("fold")
    extra = pd.read_csv(RES / experiment / "extra_fold_metrics.csv").set_index("fold")
    return shared.join(extra.drop(columns=["train_macro_f1"]), how="inner")


def main() -> None:
    sel = json.loads((RES / "step1_selection.json").read_text())
    rows = []
    for setup in ("3class", "binary"):
        s = sel[setup]
        lr = f"e1_{s['best_lr']['model']}_{setup}"
        nb = f"e2_{s['best_nb']['model']}_{setup}"
        pairs = [
            ("best LR", lr, "best NB", nb),
            ("LR L2 selected", f"e1_lr_l2_{setup}", "LR unpenalized", f"e1_lr_none_{setup}"),
            ("LR L2 selected", f"e1_lr_l2_{setup}", "LR L1 selected", f"e1_lr_l1_{setup}"),
            ("NB (b) signed log", f"e2_nb_b_signed_log_{setup}", "NB (a) standardized",
             f"e2_nb_a_standardized_{setup}"),
            ("NB (b) signed log", f"e2_nb_b_signed_log_{setup}", "NB (c) log + Spearman",
             f"e2_nb_c_log_spearman_{setup}"),
            ("best LR, 76 features", lr, "best LR, 56 features", f"e3_f56_no_centroid_lr_{setup}"),
            ("best NB, 76 features", nb, "best NB, 56 features", f"e3_f56_no_centroid_nb_{setup}"),
            ("best LR", lr, "majority dummy", f"e0_dummy_most_frequent_{setup}"),
        ]
        metrics = ["validation_macro_f1", "validation_accuracy", "balanced_accuracy", "mcc"]
        metrics += (["roc_auc", "pr_auc", "brier", "log_loss"] if setup == "binary"
                    else ["roc_auc_ovr_macro", "pr_auc_ovr_macro", "brier_multiclass", "log_loss"])
        for a_name, a_exp, b_name, b_exp in pairs:
            a, b = fold_table(a_exp), fold_table(b_exp)
            n_val = float(a["n_validation"].mean())
            n_tr = float(a["n_train"].mean())
            for metric in metrics:
                if a[metric].isna().any() or b[metric].isna().any():
                    continue  # log loss is not applicable to the dummy floor
                d = (a[metric] - b[metric]).to_numpy(dtype=float)
                row = {"setup": setup, "model_a": a_name, "experiment_a": a_exp,
                       "model_b": b_name, "experiment_b": b_exp, "metric": metric}
                row.update({f"fold{k}_diff": float(d[k]) for k in range(len(d))})
                row.update(corrected_paired_t(d, n_tr, n_val, metric))
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "paired_comparisons.csv", index=False)
    show = out.loc[out["metric"].eq("validation_macro_f1"),
                   ["setup", "model_a", "model_b", "mean_diff", "sd_diff",
                    "folds_favouring_first", "corrected_t", "corrected_p_two_sided"]]
    pd.set_option("display.width", 200)
    print(show.round(4).to_string(index=False))
    print(f"wrote {len(out)} rows to results/person1/paired_comparisons.csv")


if __name__ == "__main__":
    main()
