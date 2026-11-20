"""Step 3: label drift between catalog versions and E5 temporal validation.

Part A, label crosstab: old (Kaggle copy) vs current (NASA 2026-10-06)
koi_disposition for every KOI, and for the development partition only.

Part B, E5: train on the old catalog's CONFIRMED vs FALSE POSITIVE labels and
old feature values, then predict how the old catalog's CANDIDATE objects were
later resolved. Development-partition stars only; training stars never host
an evaluation object. Compared against an always-planet predictor and a
constant prior predictor. Sensitivity check: drop training objects whose old
label has since changed.

Two references separate the causes of any drop:
* in-snapshot: grouped 5-fold CV on the old training rows (same features),
* current values: the saved development folds of the current table with the
  same reduced feature set.

Run from the repository root after Steps 1 and 2:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python experiments/person1_baselines/run_step3_e5_temporal.py
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from exoplanet.config import GROUP_COLUMN, ID_COLUMN, ROOT
from exoplanet.development import load_development_bundle, split_identity_record
from exoplanet.models.person1_baselines.calibration import make_calibrated
from exoplanet.models.person1_baselines.cv import run_cv_many, write_experiment
from exoplanet.models.person1_baselines.data import POSITIVE, class_order, setup_frame
from exoplanet.models.person1_baselines.metrics import all_metrics
from person1_models import SEED, config_label, make_model, model_config
from exoplanet.models.person1_baselines.temporal import (
    KAGGLE_CSV,
    crosstab_summary,
    label_crosstab,
    load_versions,
    temporal_sets,
)

OUT = ROOT / "results" / "person1"
CLASSES = ("CONFIRMED", "FALSE POSITIVE")
TOP_K = (25, 50, 100, 200)


def parse_C(config: str):
    return None if config in ("C=inf", "-") else float(config.split("=")[1])


def build(model: dict, train: pd.DataFrame):
    if model.get("method"):
        return make_calibrated(partial(make_model, model["kind"], model["C"]), model["method"],
                               train[GROUP_COLUMN].to_numpy(), train["label_old"].to_numpy())
    return make_model(model["kind"], model["C"])


def proba_frame(est, X):
    P = est.predict_proba(X)
    frame = pd.DataFrame(P, columns=[str(c) for c in est.classes_], index=X.index)
    return frame[list(CLASSES)]


def evaluate(y_true, proba, pred, *, log_loss_applicable=True):
    out = all_metrics(y_true, pred, proba, CLASSES, log_loss_applicable=log_loss_applicable)
    order = np.argsort(-proba["CONFIRMED"].to_numpy(), kind="stable")
    y_sorted = np.asarray(y_true)[order]
    for k in TOP_K:
        if k <= len(y_sorted):
            out[f"precision_at_{k}"] = float((y_sorted[:k] == POSITIVE).mean())
    return out


def in_snapshot_cv(model, train, features):
    """Grouped 5-fold CV inside the old training rows (reference only)."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    y = train["label_old"]
    rows = []
    for k, (tr, va) in enumerate(splitter.split(train, y, groups=train[GROUP_COLUMN])):
        t, v = train.iloc[tr], train.iloc[va]
        est = build(model, t).fit(t[features], t["label_old"])
        proba = proba_frame(est, v[features])
        pred = est.predict(v[features]).astype(str)
        rows.append({"fold": k, **evaluate(v["label_old"], proba, pred)})
    return pd.DataFrame(rows)


def main(n_jobs: int) -> None:
    t0 = time.time()
    folder = OUT / "e5_temporal"
    folder.mkdir(parents=True, exist_ok=True)
    cross_dir = OUT / "label_crosstab"
    cross_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ Part A: label crosstab
    merged = load_versions()
    t_all = label_crosstab(merged)
    t_dev = label_crosstab(merged.loc[merged["partition"].eq("development")])
    t_all.to_csv(cross_dir / "crosstab_all_kois.csv")
    t_dev.to_csv(cross_dir / "crosstab_development_partition.csv")
    cross = {"all_kois": crosstab_summary(t_all),
             "development_partition": crosstab_summary(t_dev),
             "note": "Label counts only; no model is fit or scored here. The all-KOI table "
                     "repeats the earlier comparison; the development-partition table is what "
                     "E5 draws on.",
             "kaggle_file": "experiments/person1_baselines/data/kaggle_cumulative.csv"}
    (cross_dir / "summary.json").write_text(json.dumps(cross, indent=2))
    print("crosstab, all KOIs\n", t_all.to_string())
    print(json.dumps(cross["all_kois"], indent=1))
    print("crosstab, development partition\n", t_dev.to_string())

    # ------------------------------------------------ Part B: E5
    bundle = load_development_bundle()
    sets = temporal_sets(merged, list(bundle.feature_columns))
    F = sets["features"]
    print(f"E5 features kept {len(F)}, lost {len(sets['lost_features'])}")
    print(json.dumps(sets["counts"], indent=1))

    sel1 = json.loads((OUT / "step1_selection.json").read_text())["binary"]
    sel2 = json.loads((OUT / "e4_calibration" / "selected_calibrated_model.json").read_text())
    models = {
        f"LR {sel1['best_lr']['model']} {sel1['best_lr']['config']}":
            {"kind": sel1["best_lr"]["model"], "C": parse_C(sel1["best_lr"]["config"]), "method": None},
        "NB (b) signed log": {"kind": "nb_b_signed_log", "C": None, "method": None},
        f"{sel2['model']} {sel2['config']} + {sel2['method']} calibration":
            {"kind": sel2["model"], "C": sel2["C"], "method": sel2["method"]},
    }

    ev = sets["evaluation"]
    y_ev = ev["label_new"].astype(str)
    rows, preds = [], pd.DataFrame({ID_COLUMN: ev[ID_COLUMN], GROUP_COLUMN: ev[GROUP_COLUMN],
                                    "label_old": ev["label_old"], "label_new": y_ev})
    for train_name, train in (("all old labels", sets["train"]),
                              ("old labels unchanged since", sets["train_label_stable"])):
        prior = float(train["label_old"].eq(POSITIVE).mean())
        for name, model in models.items():
            est = build(model, train).fit(train[F], train["label_old"])
            proba = proba_frame(est, ev[F])
            pred = pd.Series(est.predict(ev[F]).astype(str), index=ev.index)
            rows.append({"training_set": train_name, "model": name, "n_train": len(train),
                         "n_evaluation": len(ev), **evaluate(y_ev, proba, pred)})
            if train_name == "all old labels":
                preds[f"p_confirmed__{name}"] = proba["CONFIRMED"].round(6)
        # Baselines
        always = pd.DataFrame({"CONFIRMED": 1.0, "FALSE POSITIVE": 0.0}, index=ev.index)
        rows.append({"training_set": train_name, "model": "always planet (predict CONFIRMED)",
                     "n_train": len(train), "n_evaluation": len(ev),
                     **evaluate(y_ev, always, pd.Series(POSITIVE, index=ev.index),
                                log_loss_applicable=False)})
        const = pd.DataFrame({"CONFIRMED": prior, "FALSE POSITIVE": 1 - prior}, index=ev.index)
        prior_pred = pd.Series(POSITIVE if prior >= 0.5 else "FALSE POSITIVE", index=ev.index)
        rows.append({"training_set": train_name,
                     "model": f"constant prior P(CONFIRMED)={prior:.3f}",
                     "n_train": len(train), "n_evaluation": len(ev),
                     **evaluate(y_ev, const, prior_pred)})
    temporal = pd.DataFrame(rows)
    temporal.insert(0, "evaluation", "old CANDIDATE resolved in current table (old feature values)")

    # In-snapshot reference
    ref_rows = []
    for name, model in models.items():
        cv = in_snapshot_cv(model, sets["train"], F)
        ref = {"evaluation": "grouped 5-fold CV inside old training rows (reference)",
               "training_set": "all old labels", "model": name}
        for col in cv.columns.drop("fold"):
            ref[col] = float(cv[col].mean())
            ref[f"{col}_sd"] = float(cv[col].std(ddof=1))
        ref_rows.append(ref)
    reference = pd.DataFrame(ref_rows)

    # Current-values reference on the saved development folds, same reduced features
    frame, y = setup_frame(bundle.frame, "binary")
    jobs = {}
    for name, model in models.items():
        if model["method"]:
            factory = partial(make_calibrated, partial(make_model, model["kind"], model["C"]),
                              model["method"])
        else:
            factory = partial(make_model, model["kind"], model["C"])
        jobs[name] = dict(frame=frame, y=y, columns=F, factory=factory, setup="binary",
                          classes=class_order("binary"))
    current = run_cv_many(jobs, n_jobs=n_jobs)
    cur_rows = []
    for name, cv in current.items():
        model = models[name]
        exp = f"e5_reference_current_values_{model['kind']}{'_' + model['method'] if model['method'] else ''}_binary"
        cfg = model_config(model["kind"], model["C"])
        cfg.update({"experiment": "E5 reference", "feature_set": "E5 reduced set (old-file columns)",
                    "calibration": model["method"]})
        write_experiment(exp, cv, cfg, save_oof=False)
        extra = json.loads((OUT / exp / "extra_metrics.json").read_text())["fold_mean_sd"]
        cur_rows.append({"evaluation": "current table, saved development folds, E5 features (reference)",
                         "training_set": "current labels", "model": name,
                         **{k: v["mean"] for k, v in extra.items()},
                         **{f"{k}_sd": v["std"] for k, v in extra.items()}, "path": f"results/person1/{exp}"})
    current_df = pd.DataFrame(cur_rows)

    temporal.to_csv(folder / "temporal_metrics.csv", index=False)
    pd.concat([reference, current_df], ignore_index=True).to_csv(folder / "reference_metrics.csv",
                                                                 index=False)
    preds.to_csv(folder / "evaluation_predictions.csv", index=False)
    (folder / "features.json").write_text(json.dumps(
        {"kept": F, "lost": sets["lost_features"], "n_kept": len(F),
         "n_lost": len(sets["lost_features"])}, indent=2))
    (folder / "counts.json").write_text(json.dumps(sets["counts"], indent=2))
    config = {
        "experiment": "E5 temporal validation",
        "train": "old catalog CONFIRMED vs FALSE POSITIVE labels and old feature values, "
                 "development-partition stars, measured old koi_depth, stars hosting an "
                 "evaluation object removed",
        "evaluation": "old catalog CANDIDATE objects now CONFIRMED or FALSE POSITIVE in the "
                      "current table, old feature values, development-partition stars",
        "sensitivity": "training rows whose old label differs from the current label removed",
        "models": models,
        "threshold": 0.5,
        "kaggle_file": "experiments/person1_baselines/data/kaggle_cumulative.csv",
        "kaggle_sha256": sets["kaggle_sha256"],
        "split": split_identity_record(),
        "test_scored": False,
    }
    (folder / "model_config.json").write_text(json.dumps(config, indent=2, default=str))

    pd.set_option("display.width", 220)
    show = ["training_set", "model", "macro_f1", "accuracy", "recall_CONFIRMED",
            "recall_FALSE_POSITIVE", "roc_auc", "pr_auc", "brier", "precision_at_100"]
    print(temporal[show].round(4).to_string(index=False))
    print(reference[["model", "macro_f1", "roc_auc", "pr_auc", "brier"]].round(4).to_string(index=False))
    print(current_df[["model", "macro_f1", "roc_auc", "pr_auc", "brier"]].round(4).to_string(index=False))
    print(f"runtime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=-1)
    main(parser.parse_args().n_jobs)
