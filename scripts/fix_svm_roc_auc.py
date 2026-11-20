"""Logged fix for the SVM ROC-AUC value that run_final_test.py recorded as NaN.

Only this one metric is computed. Steps:
  1. Refit RBF SVM and Linear SVM with the same factories on the 7,381
     development rows, as run_final_test.py does.
  2. Identity check: test predictions must equal the saved columns in
     results/final_test/test_predictions.csv on all 1,820 rows. If any row
     differs, stop before computing anything.
  3. ROC-AUC = macro mean of per-class one-vs-rest binary ROC-AUC, using each
     class's decision_function column (matched to pipe.classes_).
  4. Validation: the same formula on the five predict_proba models must match
     metrics_summary.csv to 4 decimals.
Writes results/final_test/svm_roc_auc_fix.csv. Does not edit any other result.

Run:  .venv/bin/python scripts/fix_svm_roc_auc.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_final_test as rft  # noqa: E402

OUT = rft.OUT
SVMS = ["RBF SVM", "Linear SVM"]
PROBA_MODELS = ["Random Forest", "Pruned Decision Tree", "Logistic Regression", "KNN", "Naive Bayes"]


def per_class_macro_auc(pipe, X, y) -> float:
    """Macro mean of one-vs-rest binary ROC-AUC, one score column per class."""
    _, scores = rft.get_scores(pipe, X)
    classes = list(pipe.classes_)
    values = [roc_auc_score((y == c).astype(int), scores[:, classes.index(c)]) for c in rft.CLASS_ORDER]
    return float(np.mean(values))


def build(name, dev, test, columns):
    if name in rft.SAVED_MODELS:
        return rft.load_saved(rft.SAVED_MODELS[name])["pipeline"]
    pipe = rft.FACTORIES[name]()
    pipe.fit(dev[columns], dev[rft.TARGET_COLUMN])
    return pipe


def main() -> int:
    columns = rft.feature_list()
    dev = rft.load_matrix("development")
    test = rft.load_matrix("test")
    assert len(dev) == rft.N_DEV and len(test) == rft.N_TEST
    X, y = test[columns], test[rft.TARGET_COLUMN]

    saved = pd.read_csv(OUT / "test_predictions.csv")
    assert saved[rft.ID_COLUMN].tolist() == test[rft.ID_COLUMN].tolist(), "row order differs"

    pipes, identity = {}, {}
    for name in SVMS:
        pipes[name] = build(name, dev, test, columns)
        identity[name] = int((pipes[name].predict(X) != saved[name].to_numpy()).sum())
    print("identity check, rows that differ from test_predictions.csv:", identity, "of", len(test))
    if any(identity.values()):
        print("IDENTITY CHECK FAILED. Nothing was computed.")
        return 1

    values = {n: per_class_macro_auc(pipes[n], X, y) for n in SVMS}

    original = pd.read_csv(OUT / "metrics_summary.csv").set_index("model")["roc_auc_ovr_macro"]
    ok = True
    print(f"{'model':<22}{'per-class':>12}{'recorded':>12}  match 4dp")
    for name in PROBA_MODELS:
        got = per_class_macro_auc(build(name, dev, test, columns), X, y)
        recorded = float(original[name])
        match = round(got, 4) == round(recorded, 4)
        ok &= match
        print(f"{name:<22}{got:>12.6f}{recorded:>12.6f}  {match}")
    print("validation:", "PASSED" if ok else "FAILED")
    if not ok:
        print("Not writing the fix file.")
        return 1

    pd.DataFrame([{"model": n, "roc_auc_ovr_macro": values[n], "score_type": "decision_function",
                   "method": "macro mean of per-class one-vs-rest binary ROC-AUC",
                   "rows_differing_from_original_predictions": identity[n]} for n in SVMS]
                 ).to_csv(OUT / "svm_roc_auc_fix.csv", index=False)
    for n in SVMS:
        print(f"{n}: {values[n]:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
