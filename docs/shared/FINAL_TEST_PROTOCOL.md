# Final Test Protocol

This file is written before any test score exists. It fixes how the test partition is scored.

## 1. Approval record

Team approval: [docs/shared/meeting_notes/2026-11-20_final_plan_approval.md](meeting_notes/2026-11-20_final_plan_approval.md)

## 2. Data

- Dataset SHA-256: `a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab`
- Split SHA-256: `d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a`
- Random seed: 245
- Features: 76, from `data/processed/proposed_leakage_aware_features.csv`
- Split file: `data/processed/proposed_splits.csv` (host-star grouped by `kepid`)

## 3. Training data

All 7,381 eligible development rows (`partition = development` and `primary_supervised_row = True`).

## 4. Test data

Test rows that have a measured transit depth.

Row count, taken from `proposed_splits.csv` and `incomplete_koi_split_assignments.csv` using only the `partition` and `primary_supervised_row` columns (no feature values were loaded):

| Item | Rows |
| --- | --- |
| Test rows in total | 1,911 |
| Test rows without transit depth (listed in `incomplete_koi_split_assignments.csv`) | 91 |
| **Test rows scored** | **1,820** |

The test script must check that it loads exactly 1,820 rows and stop if the count differs.

## 5. Models and frozen settings

Seven models from the approval note, plus a floor baseline.

| Model | Frozen settings |
| --- | --- |
| Random Forest | 100 trees, max_depth 20, max_features sqrt, min_samples_leaf 2, class_weight balanced |
| RBF SVM | C=10, gamma scale, class_weight balanced |
| Linear SVM | C=10, class_weight balanced |
| Logistic Regression | L2, C=1000, class_weight balanced |
| Pruned Decision Tree | ccp_alpha=0.0011107571059714625 |
| KNN | k=7, euclidean, distance weighting |
| Naive Bayes | signed-log transform |
| Majority-class baseline (floor) | Predicts the most common development class |

### Saved pipelines (Random Forest and pruned Decision Tree)

These two existing saved pipelines are scored as they are:

- `results/person3/random_forest/tuned_forest.joblib`
- `results/person3/decision_tree/stage3/selected_development_tree.joblib`

Metadata check (read from the JSON files in `results/person3`, no model was loaded):

- Both metadata files record partition `development`, 7,381 rows, 76 features, seed 245, the same split SHA-256, and `test_scored: false`.
- Both have a `development_fit` block with role `in_sample_development_fit`.
- The Random Forest settings in the metadata match the frozen settings (100 trees, max_depth 20, max_features sqrt, min_samples_leaf 2, class_weight balanced, random_state 245).
- The Decision Tree settings in the metadata match the frozen setting (ccp_alpha 0.0011107571059714625, min_samples_split 2, min_samples_leaf 1, no max_depth, random_state 245).
- The metadata does not state the row count used to fit the saved `.joblib` files themselves. The numbers are consistent with a fit on all 7,381 development rows, but the metadata alone does not prove it. Section 6 closes this gap.

### Other five models

RBF SVM, Linear SVM, Logistic Regression, KNN and Naive Bayes are fit fresh on the 7,381 development rows. Each model is built with the exact factory or pipeline code that produced its development result. No new model code is written for the test. Preprocessing is fit on the development rows only and then applied to the test rows.

Factory files:

- Logistic Regression and Naive Bayes: `experiments/person1_baselines/person1_models.py` (`make_model`), with the transforms in `src/exoplanet/models/person1_baselines/transforms.py`.
- KNN, Linear SVM and RBF SVM: `src/exoplanet/models/person2_knn_svm/models.py` (`make_knn`, `make_linear_svm`, `make_rbf_svm`).

Shared preprocessing is `make_preprocess_pipeline` in `src/exoplanet/preprocess.py`. It is a `SimpleImputer(strategy="median", keep_empty_features=True)`, followed by a `StandardScaler()` when `scale=True`. All five models use `scale=True`.

| Model | Factory call | Estimator class | Non-default parameters | Steps in order |
| --- | --- | --- | --- | --- |
| Logistic Regression | `make_model("lr_l2", C=1000.0)` | `LogisticRegression` | `C=1000.0`, `l1_ratio=0.0` (L2), `solver="lbfgs"`, `class_weight="balanced"`, `max_iter=20000`, `tol=1e-4` | median impute, standardize, LogisticRegression |
| Naive Bayes | `make_model("nb_b_signed_log")` | `GaussianNB` | none (`var_smoothing=1e-9` is the default) | median impute, signed log1p (`FunctionTransformer(signed_log1p)`), standardize, GaussianNB |
| KNN | `make_knn(n_neighbors=7, metric="euclidean", weights="distance")` | `KNeighborsClassifier` | `n_neighbors=7`, `weights="distance"` (`metric="euclidean"` is passed and equals the default Minkowski with p=2) | median impute, standardize, KNN |
| Linear SVM | `make_linear_svm(C=10.0, class_weight="balanced")` | `LinearSVC` | `C=10.0`, `class_weight="balanced"`, `max_iter=10000` (set by the factory) | median impute, standardize, LinearSVC |
| RBF SVM | `make_rbf_svm(C=10.0, gamma="scale", class_weight="balanced")` | `SVC` | `kernel="rbf"`, `C=10.0`, `class_weight="balanced"` (`gamma="scale"` is the default) | median impute, standardize, SVC |

`signed_log1p(x)` is `sign(x) * log(1 + |x|)`. The majority-class baseline is `make_model("dummy_most_frequent")`: a `DummyClassifier(strategy="most_frequent")` after median imputation with no scaling.

## 6. Pre-test verification (must pass before any test row is loaded)

The test script runs these checks first. No test row may be read until all pass.

1. **Decision Tree.** Load the saved pipeline. Assert that the tree's root node sample count `tree_.n_node_samples[0]` equals 7381 and that `n_features_in_` equals 76.
2. **Random Forest.** Refit a forest with the frozen settings and `random_state` 245 on the 7,381 development rows. Assert that its predictions on those same development rows match the saved forest exactly. Also assert that the saved forest's `n_features_in_` equals 76.
3. **If either check fails, stop and report. Do not score the test set.**

This step is only written here. It has not been run.

## 7. Feature configurations

Configuration A (76 features) is the only configuration scored on the test set. Configuration B (56 features, no centroid columns) stays a development-fold result and is not scored on the test set.

## 8. Primary metric and expectation

- Primary metric: macro-F1.
- Random Forest is the expected best model. This is stated before the test.

## 9. Metrics to report

For every model:

- Accuracy and balanced accuracy
- Macro precision, recall and F1
- Per-class precision, recall and F1
- Matthews correlation coefficient (MCC)
- Confusion matrix
- One-vs-rest macro ROC-AUC, computed the same way for every model (`roc_auc_score` with `multi_class="ovr"`, `average="macro"`). Score type rule: use `predict_proba` when the pipeline provides it. For SVMs, use `decision_function` scores, not `probability=True`. The score type used for each model is recorded in the results:

  | Model | Score type |
  | --- | --- |
  | Random Forest | `predict_proba` |
  | Pruned Decision Tree | `predict_proba` |
  | Logistic Regression | `predict_proba` |
  | Naive Bayes | `predict_proba` |
  | KNN | `predict_proba` |
  | RBF SVM | `decision_function` |
  | Linear SVM | `decision_function` |

  The majority-class baseline is excluded from ROC-AUC.
- Development macro-F1 next to test macro-F1. The development value is the mean over the 5 grouped development folds, read from each model's saved development summary. It is not recomputed:

  | Model | File |
  | --- | --- |
  | Random Forest | `results/person3/random_forest/model_config.json` (`mean_validation_macro_f1` = 0.8156171986031888) |
  | Pruned Decision Tree | `results/person3/decision_tree/stage3/stage3_metrics.json` (`selected.mean_validation_macro_f1` = 0.7716) |
  | RBF SVM | `results/person2/rbf_svm_tuned/summary.json` |
  | Linear SVM | `results/person2/linear_svm_tuned/summary.json` |
  | KNN | `results/person2/knn_tuned/summary.json` |
  | Logistic Regression | `results/person1/e1_lr_l2_3class/summary.json` |
  | Naive Bayes | `results/person1/e2_nb_b_signed_log_3class/summary.json` |
  | Majority-class baseline | `results/person1/e0_dummy_most_frequent_3class/summary.json` |

  Do not use `tuned_pooled_macro_f1` from `results/person3/random_forest/stage4_metrics.json` for Random Forest. It is a pooled value, not the 5-fold mean.

## 10. Uncertainty

- 95% bootstrap confidence interval for macro-F1 and for accuracy. Resample host stars (`kepid`), 1,000 resamples, seed 245.
- Paired McNemar test on the same test rows for: Random Forest vs RBF SVM, and Random Forest vs Logistic Regression. The test is exact (binomial), two-sided, on per-row correct vs incorrect predictions. The majority-class baseline is excluded.

## 11. Run record

The script saves:

- Git commit hash
- Dataset SHA-256 and split SHA-256
- Library versions from `requirements.txt`
- Timestamp

## 12. Output folder

`results/final_test/`

## 13. Rerun rule

The test is run once. If the run crashes, only the bug is fixed and the rerun is logged. No change is made to models, settings or features after any test score is seen. Results are reported as they come out.

## 14. Who runs it

Danish Waseem, after this pull request is merged.
