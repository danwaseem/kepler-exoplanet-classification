#!/usr/bin/env python3
"""Person 2: score the default and tuned KNN / SVM models with the shared helpers.

Run from the repository root:

    python experiments/person2_knn_svm/run_final_models.py

Writes one folder per experiment under results/person2/<experiment_name>/
(fold_metrics.csv, summary.json, per_class_metrics.csv,
oof_confusion_matrix.csv, model_config.json) and a combined
results/person2/summary.csv. Development folds only; the test partition is
not scored.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from exoplanet.config import CLASS_ORDER, DATA_PROCESSED
from exoplanet.development import (
    development_fold_masks,
    feature_matrix,
    labels,
    load_development_bundle,
)
from exoplanet.evaluation import cross_validate_estimator, write_experiment_results
from exoplanet.models.person2_knn_svm import models as m


def centroid_columns() -> list[str]:
    table = pd.read_csv(DATA_PROCESSED / "proposed_leakage_aware_features.csv")
    return table.loc[table["provisional"].astype(bool), "column"].tolist()


def run(bundle, name, factory_fn, config, *, configuration, description):
    """Grouped 5-fold CV with the shared helper, plus out-of-fold ROC-AUC."""
    fitted = []

    def factory():
        estimator = factory_fn()
        fitted.append(estimator)
        return estimator

    start = time.time()
    summary = cross_validate_estimator(bundle, factory)

    # One-vs-rest macro ROC-AUC from the same fitted fold models
    frame = bundle.frame
    x = feature_matrix(frame, bundle.feature_columns)
    y = labels(frame).to_numpy()
    scores = np.zeros((len(frame), len(CLASS_ORDER)))
    for fold_id, estimator in enumerate(fitted):
        _, valid = development_fold_masks(frame, fold_id)
        model = estimator.named_steps["model"]
        raw = (estimator.predict_proba(x.loc[valid]) if hasattr(model, "predict_proba")
               else estimator.decision_function(x.loc[valid]))
        order = [list(estimator.classes_).index(c) for c in CLASS_ORDER]
        scores[valid.to_numpy()] = raw[:, order]
    summary["oof_roc_auc_ovr_macro"] = float(
        np.mean([roc_auc_score(y == c, scores[:, i]) for i, c in enumerate(CLASS_ORDER)])
    )

    full_config = {
        **config,
        "configuration": configuration,
        "description": description,
        "n_features": len(bundle.feature_columns),
        "imputation": "median, fit on training fold",
        "selection_metric": "grouped 5-fold validation macro-F1",
    }
    out = write_experiment_results("person2", name, summary, full_config)
    print(f"{name:40s} macro-F1 {summary['mean_validation_macro_f1']:.4f} "
          f"± {summary['std_validation_macro_f1']:.4f}  AUC {summary['oof_roc_auc_ovr_macro']:.3f}  "
          f"({time.time() - start:.0f}s) -> {out.relative_to(ROOT)}", flush=True)

    per_class = {row["class"]: row for row in summary["per_class"]}
    return {
        "experiment": name,
        "model": config["model"],
        "configuration": configuration,
        "description": description,
        "standardized": config["standardized"],
        "n_features": len(bundle.feature_columns),
        "hyperparameters": str(config["hyperparameters"]),
        "mean_validation_macro_f1": summary["mean_validation_macro_f1"],
        "std_validation_macro_f1": summary["std_validation_macro_f1"],
        "mean_validation_accuracy": summary["mean_validation_accuracy"],
        "std_validation_accuracy": summary["std_validation_accuracy"],
        "mean_train_macro_f1": summary["mean_train_macro_f1"],
        "mean_macro_f1_gap": summary["mean_macro_f1_gap"],
        "oof_roc_auc_ovr_macro": summary["oof_roc_auc_ovr_macro"],
        "oof_recall_confirmed": per_class["CONFIRMED"]["recall"],
        "oof_recall_candidate": per_class["CANDIDATE"]["recall"],
        "oof_recall_false_positive": per_class["FALSE POSITIVE"]["recall"],
        "test_scored": False,
    }


def main() -> None:
    bundle_a = load_development_bundle()
    drop = set(centroid_columns())
    bundle_b = dataclasses.replace(
        bundle_a, feature_columns=[c for c in bundle_a.feature_columns if c not in drop]
    )
    assert len(bundle_b.feature_columns) == 56

    A = ("A", "primary scientific set (76 features)")
    B = ("B", "without provisional centroid offsets (56 features), retuned")
    B_FIXED = ("B", "without provisional centroid offsets (56 features), configuration-A hyperparameters")

    experiments = [
        # Configuration A: scaling experiment + default vs tuned
        (bundle_a, "knn_default_unscaled", lambda: m.make_knn(scale=False, **m.KNN_DEFAULT), "KNN", False, m.KNN_DEFAULT, A),
        (bundle_a, "knn_default_scaled", lambda: m.make_knn(**m.KNN_DEFAULT), "KNN", True, m.KNN_DEFAULT, A),
        (bundle_a, "knn_tuned", lambda: m.make_knn(**m.KNN_TUNED_A), "KNN", True, m.KNN_TUNED_A, A),
        (bundle_a, "linear_svm_default", lambda: m.make_linear_svm(**m.LINEAR_SVM_DEFAULT), "Linear SVM", True, m.LINEAR_SVM_DEFAULT, A),
        (bundle_a, "linear_svm_tuned", lambda: m.make_linear_svm(**m.LINEAR_SVM_TUNED), "Linear SVM", True, m.LINEAR_SVM_TUNED, A),
        (bundle_a, "rbf_svm_default_unscaled", lambda: m.make_rbf_svm(scale=False, **m.RBF_SVM_DEFAULT), "RBF SVM", False, m.RBF_SVM_DEFAULT, A),
        (bundle_a, "rbf_svm_default_scaled", lambda: m.make_rbf_svm(**m.RBF_SVM_DEFAULT), "RBF SVM", True, m.RBF_SVM_DEFAULT, A),
        (bundle_a, "rbf_svm_tuned", lambda: m.make_rbf_svm(**m.RBF_SVM_TUNED_A), "RBF SVM", True, m.RBF_SVM_TUNED_A, A),
        # Configuration B: centroid sensitivity, retuned on 56 features
        (bundle_b, "knn_tuned_no_centroid", lambda: m.make_knn(**m.KNN_TUNED_B), "KNN", True, m.KNN_TUNED_B, B),
        (bundle_b, "linear_svm_tuned_no_centroid", lambda: m.make_linear_svm(**m.LINEAR_SVM_TUNED), "Linear SVM", True, m.LINEAR_SVM_TUNED, B),
        (bundle_b, "rbf_svm_tuned_no_centroid", lambda: m.make_rbf_svm(**m.RBF_SVM_TUNED_B), "RBF SVM", True, m.RBF_SVM_TUNED_B, B),
        # Configuration B with configuration-A hyperparameters (same approach as results/person3/centroid_sensitivity)
        (bundle_b, "knn_tuned_no_centroid_fixed_params", lambda: m.make_knn(**m.KNN_TUNED_A), "KNN", True, m.KNN_TUNED_A, B_FIXED),
        (bundle_b, "rbf_svm_tuned_no_centroid_fixed_params", lambda: m.make_rbf_svm(**m.RBF_SVM_TUNED_A), "RBF SVM", True, m.RBF_SVM_TUNED_A, B_FIXED),
    ]

    rows = []
    for bundle, name, factory_fn, model_name, scaled, params, (cfg, desc) in experiments:
        config = {"model": model_name, "standardized": scaled, "hyperparameters": params}
        rows.append(run(bundle, name, factory_fn, config, configuration=cfg, description=desc))

    table = pd.DataFrame(rows)
    out = ROOT / "results" / "person2" / "summary.csv"
    table.to_csv(out, index=False)
    print(f"\nWrote {out.relative_to(ROOT)}")
    print(table[["experiment", "mean_validation_macro_f1", "std_validation_macro_f1",
                 "oof_roc_auc_ovr_macro", "oof_recall_candidate"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
