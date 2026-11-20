# Deviation log

Date: 2026-11-20

## What happened

The original test run (`scripts/run_final_test.py --run`, code commit `b3cbb6789052af9899704280c0817ac2abe39bb3`) recorded NaN for the ROC-AUC of RBF SVM and Linear SVM. This was a script bug. The script used a library call that does not accept decision scores for multiclass one-vs-rest, and it silently stored NaN when the call failed.

## The fix

The fix computed only this one metric, after the run. It is in `scripts/fix_svm_roc_auc.py`. The result is in `svm_roc_auc_fix.csv`.

- Both SVMs were refit with the same factories on the 7,381 development rows, exactly as the original run did.
- Their test predictions were verified identical to the saved columns in `test_predictions.csv` on all 1,820 rows. Zero rows differ.
- ROC-AUC is the macro mean of the per-class one-vs-rest binary ROC-AUC. Each class uses its own column of the `decision_function` scores.
- SVM ROC-AUC uses decision scores, not probabilities.
- The same per-class formula was applied to the five `predict_proba` models. It matched their existing values in `metrics_summary.csv` to 4 decimals.

## What did not change

No model, setting, feature, or other metric changed. `metrics_summary.csv` is not edited. The original NaN values stay in it as the record of the first run.
