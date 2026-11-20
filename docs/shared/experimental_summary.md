# Final team summary

This is the final summary of the DATA 245 Kepler KOI classification project. Every number below was read from the file named in its section. Nothing here is a new result. Stage-level detail for the Decision Tree and Random Forest work is in `docs/person3/` and in the saved results under `results/`.

Related records: [final test protocol](FINAL_TEST_PROTOCOL.md), [team approval record](meeting_notes/2026-11-20_final_plan_approval.md), [deviation log](../../results/final_test/DEVIATION_LOG.md).

## a. Setup

Sources: `results/final_test/run_record.json` and `data/processed/proposed_splits.csv`.

- Dataset SHA-256: `a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab`
- Split SHA-256: `d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a`
- Development rows used for training and cross-validation: 7,381
- Test rows scored: 1,820
- Features: 76
- Development folds: 5 grouped by host star (`kepid`)
- Random seed: 245

## b. Development results (configuration A)

Source: `results/shared/combined_comparison/summary_config_a.csv`. Mean and standard deviation over the 5 grouped development folds. Gap is training macro-F1 minus validation macro-F1.

| Model | Macro-F1 | Accuracy | Macro precision | Macro recall | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random forest (tuned) | 0.8156 ± 0.0130 | 0.8433 ± 0.0105 | 0.8114 ± 0.0133 | 0.8216 ± 0.0133 | 0.1702 ± 0.0130 |
| RBF SVM | 0.7888 ± 0.0116 | 0.8160 ± 0.0081 | 0.7832 ± 0.0111 | 0.7983 ± 0.0133 | 0.0940 ± 0.0153 |
| Linear SVM | 0.7750 ± 0.0122 | 0.8083 ± 0.0093 | 0.7700 ± 0.0118 | 0.7842 ± 0.0115 | 0.0119 ± 0.0142 |
| Logistic regression | 0.7721 ± 0.0137 | 0.7923 ± 0.0123 | 0.7684 ± 0.0135 | 0.7905 ± 0.0142 | 0.0151 ± 0.0158 |
| Pruned DT | 0.7716 ± 0.0146 | 0.8137 ± 0.0098 | 0.7773 ± 0.0126 | 0.7697 ± 0.0156 | 0.0468 ± 0.0202 |
| KNN | 0.7364 ± 0.0146 | 0.7811 ± 0.0131 | 0.7401 ± 0.0152 | 0.7438 ± 0.0127 | 0.2636 ± 0.0146 |
| Naive Bayes | 0.6699 ± 0.0205 | 0.6994 ± 0.0119 | 0.6817 ± 0.0220 | 0.6975 ± 0.0229 | 0.0030 ± 0.0168 |
| Majority floor | 0.2218 ± 0.0006 | 0.4984 ± 0.0019 | 0.1661 ± 0.0006 | 0.3333 ± 0.0000 | 0.0000 ± 0.0007 |
| PCA probe: original 76 | 0.7735 ± 0.0108 | 0.7950 ± 0.0085 | 0.7700 ± 0.0112 | 0.7905 ± 0.0122 | 0.0145 ± 0.0122 |
| PCA probe: PCA 25 | 0.7664 ± 0.0129 | 0.7874 ± 0.0106 | 0.7633 ± 0.0126 | 0.7852 ± 0.0142 | 0.0030 ± 0.0142 |

The two PCA rows are an information probe by Person 4, not competing models. The majority floor predicts the most common class.

## c. Paired development comparisons

Source: `results/shared/combined_comparison/paired_folds.csv`. Difference is model A minus model B on the same 5 folds. The p-value is the file's corrected resampled t-test (`nb_two_sided_p`). With 5 folds these are descriptive, not final tests.

| Model A | Model B | Metric | Mean diff | Min | Max | Folds A higher | p |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Random forest (tuned) | RBF SVM | validation_macro_f1 | +0.02683 | +0.01918 | +0.04042 | 5 of 5 | 0.0090 |
| Random forest (tuned) | RBF SVM | validation_accuracy | +0.02723 | +0.02161 | +0.03567 | 5 of 5 | 0.0031 |
| Random forest (tuned) | Logistic regression | validation_macro_f1 | +0.04357 | +0.03437 | +0.05873 | 5 of 5 | 0.0023 |
| Random forest (tuned) | Logistic regression | validation_accuracy | +0.05092 | +0.04254 | +0.06326 | 5 of 5 | 0.0010 |
| RBF SVM | Logistic regression | validation_macro_f1 | +0.01673 | +0.01085 | +0.01934 | 5 of 5 | 0.0020 |
| RBF SVM | Logistic regression | validation_accuracy | +0.02369 | +0.01777 | +0.02836 | 5 of 5 | 0.0014 |
| Pruned DT | Logistic regression | validation_macro_f1 | -0.00044 | -0.01424 | +0.01532 | 3 of 5 | 0.9572 |
| Pruned DT | Logistic regression | validation_accuracy | +0.02139 | +0.01350 | +0.02894 | 5 of 5 | 0.0093 |

## d. Centroid sensitivity (A vs B)

Source: `results/shared/combined_comparison/centroid_a_vs_b.csv`. Configuration A uses the 76 features. Configuration B drops the 20 centroid columns (10 centroid offsets and their 10 uncertainty columns) and uses 56 features. Both use development folds only.

| Model | B settings | Macro-F1 A | Macro-F1 B | B minus A | Folds where B is lower |
| --- | --- | ---: | ---: | ---: | ---: |
| Logistic regression | same as A | 0.7721 | 0.7192 | -0.0528 | 5 of 5 |
| Naive Bayes | same as A | 0.6699 | 0.6044 | -0.0656 | 5 of 5 |
| KNN | same as A | 0.7364 | 0.6301 | -0.1063 | 5 of 5 |
| Linear SVM | same as A (retuning chose the A settings: C=10, balanced) | 0.7750 | 0.7069 | -0.0681 | 5 of 5 |
| RBF SVM | same as A | 0.7888 | 0.7296 | -0.0592 | 5 of 5 |
| Pruned DT | same as A | 0.7716 | 0.7203 | -0.0513 | 5 of 5 |
| Random forest (tuned) | same as A | 0.8156 | 0.7634 | -0.0522 | 5 of 5 |

Every model is lower without the 20 centroid columns on all 5 folds. The drop in macro-F1 is 0.051 to 0.106.

These columns are Kepler Data Validation diagnostics. So the models reproduce the archive disposition using vetting-diagnostic measurements, not transit data alone.

A second table, `results/shared/combined_comparison/centroid_p2_retuned_vs_fixed.csv`, compares Person 2's retuned and fixed-settings values for reference.

## e. Final test results

Sources: `results/final_test/metrics_summary.csv`, `results/final_test/bootstrap_ci.csv`, `results/final_test/mcnemar.csv`, `results/final_test/per_class_metrics.csv` and `results/final_test/svm_roc_auc_fix.csv`. The test partition was scored once, on 1,820 rows, with configuration A only.

### Metrics

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro-F1 | MCC | ROC-AUC (OvR macro) | Score type |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Random Forest | 0.8467 | 0.8249 | 0.8159 | 0.8249 | 0.8199 | 0.7563 | 0.9519 | predict_proba |
| RBF SVM | 0.8258 | 0.8045 | 0.7916 | 0.8045 | 0.7967 | 0.7256 | 0.9249 * | decision_function |
| Linear SVM | 0.8016 | 0.7720 | 0.7606 | 0.7720 | 0.7653 | 0.6862 | 0.9114 * | decision_function |
| Logistic Regression | 0.7967 | 0.7890 | 0.7707 | 0.7890 | 0.7735 | 0.6898 | 0.9210 | predict_proba |
| Pruned Decision Tree | 0.8137 | 0.7598 | 0.7817 | 0.7598 | 0.7677 | 0.6966 | 0.9227 | predict_proba |
| KNN | 0.7753 | 0.7376 | 0.7318 | 0.7376 | 0.7291 | 0.6455 | 0.9004 | predict_proba |
| Naive Bayes | 0.7049 | 0.6981 | 0.6812 | 0.6981 | 0.6727 | 0.5683 | 0.8897 | predict_proba |
| Majority baseline | 0.4962 | 0.3333 | 0.1654 | 0.3333 | 0.2211 | 0.0000 | excluded | excluded |

\* The original run recorded NaN for the two SVM ROC-AUC values because of a script bug. The values shown come from `svm_roc_auc_fix.csv`, computed after the run from decision scores. Predictions were verified identical on all 1,820 rows. See `results/final_test/DEVIATION_LOG.md`.

### 95% bootstrap confidence intervals

Host stars (`kepid`) were resampled, 1,000 resamples, seed 245. Source: `results/final_test/bootstrap_ci.csv`.

| Model | Macro-F1 | 95% CI | Accuracy | 95% CI |
| --- | ---: | --- | ---: | --- |
| Random Forest | 0.8199 | 0.7997 to 0.8385 | 0.8467 | 0.8292 to 0.8633 |
| RBF SVM | 0.7967 | 0.7776 to 0.8155 | 0.8258 | 0.8090 to 0.8427 |
| Linear SVM | 0.7653 | 0.7424 to 0.7863 | 0.8016 | 0.7816 to 0.8213 |
| Logistic Regression | 0.7735 | 0.7531 to 0.7947 | 0.7967 | 0.7783 to 0.8168 |
| Pruned Decision Tree | 0.7677 | 0.7459 to 0.7881 | 0.8137 | 0.7952 to 0.8310 |
| KNN | 0.7291 | 0.7070 to 0.7525 | 0.7753 | 0.7572 to 0.7944 |
| Naive Bayes | 0.6727 | 0.6507 to 0.6936 | 0.7049 | 0.6843 to 0.7258 |
| Majority baseline | 0.2211 | 0.2129 to 0.2290 | 0.4962 | 0.4692 to 0.5231 |

### McNemar test

Exact binomial, two-sided, on per-row correct vs incorrect predictions. Source: `results/final_test/mcnemar.csv`.

| Model A | Model B | A right, B wrong | A wrong, B right | Discordant rows | p |
| --- | --- | ---: | ---: | ---: | ---: |
| Random Forest | RBF SVM | 112 | 74 | 186 | 0.00651 |
| Random Forest | Logistic Regression | 143 | 52 | 195 | 5.16e-11 |

### Per-class precision, recall and F1

Source: `results/final_test/per_class_metrics.csv`. Test support: CONFIRMED 549, CANDIDATE 368, FALSE POSITIVE 903.

| Model | Class | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| Random Forest | CONFIRMED | 0.8929 | 0.8962 | 0.8945 |
| Random Forest | CANDIDATE | 0.6427 | 0.7038 | 0.6719 |
| Random Forest | FALSE POSITIVE | 0.9122 | 0.8749 | 0.8932 |
| RBF SVM | CONFIRMED | 0.8688 | 0.8925 | 0.8805 |
| RBF SVM | CANDIDATE | 0.5919 | 0.6739 | 0.6302 |
| RBF SVM | FALSE POSITIVE | 0.9140 | 0.8472 | 0.8793 |
| Linear SVM | CONFIRMED | 0.8294 | 0.9035 | 0.8649 |
| Linear SVM | CANDIDATE | 0.5599 | 0.5842 | 0.5718 |
| Linear SVM | FALSE POSITIVE | 0.8926 | 0.8283 | 0.8593 |
| Logistic Regression | CONFIRMED | 0.8633 | 0.8743 | 0.8688 |
| Logistic Regression | CANDIDATE | 0.5242 | 0.7065 | 0.6019 |
| Logistic Regression | FALSE POSITIVE | 0.9245 | 0.7863 | 0.8498 |
| Pruned Decision Tree | CONFIRMED | 0.8585 | 0.8506 | 0.8545 |
| Pruned Decision Tree | CANDIDATE | 0.6485 | 0.5163 | 0.5749 |
| Pruned Decision Tree | FALSE POSITIVE | 0.8383 | 0.9125 | 0.8738 |
| KNN | CONFIRMED | 0.7522 | 0.9399 | 0.8356 |
| KNN | CANDIDATE | 0.5757 | 0.4755 | 0.5208 |
| KNN | FALSE POSITIVE | 0.8675 | 0.7973 | 0.8309 |
| Naive Bayes | CONFIRMED | 0.7171 | 0.9417 | 0.8142 |
| Naive Bayes | CANDIDATE | 0.3929 | 0.5136 | 0.4452 |
| Naive Bayes | FALSE POSITIVE | 0.9337 | 0.6390 | 0.7587 |
| Majority baseline | CONFIRMED | 0.0000 | 0.0000 | 0.0000 |
| Majority baseline | CANDIDATE | 0.0000 | 0.0000 | 0.0000 |
| Majority baseline | FALSE POSITIVE | 0.4962 | 1.0000 | 0.6632 |

## f. Development vs test macro-F1

Source: `results/final_test/metrics_summary.csv`. The development value is the mean of the 5 grouped development folds from each model's saved development summary. The file column `dev_source` names that file for each model.

| Model | Development | Test | Test minus development |
| --- | ---: | ---: | ---: |
| Random Forest | 0.8156 | 0.8199 | +0.0042 |
| RBF SVM | 0.7888 | 0.7967 | +0.0079 |
| Linear SVM | 0.7750 | 0.7653 | -0.0097 |
| Logistic Regression | 0.7721 | 0.7735 | +0.0014 |
| Pruned Decision Tree | 0.7716 | 0.7677 | -0.0039 |
| KNN | 0.7364 | 0.7291 | -0.0073 |
| Naive Bayes | 0.6699 | 0.6727 | +0.0028 |
| Majority baseline | 0.2218 | 0.2211 | -0.0007 |

## g. Label changes between catalog versions

Source: `results/person1/label_crosstab/summary.json` (all KOIs). The table compares the older catalog labels with the labels in this project's dataset.

- 925 labels changed among 9,564 KOIs.
- 591 of these were old candidates that were resolved: 443 to CONFIRMED and 148 to FALSE POSITIVE.
- The other 334 changes went in other directions: 320 FALSE POSITIVE to CANDIDATE, 13 FALSE POSITIVE to CONFIRMED, and 1 CONFIRMED to FALSE POSITIVE.

## h. Limitations

- CANDIDATE is not a physical class. It means the archive has not yet confirmed or rejected the object. On the test set, CANDIDATE has the lowest F1 of the three classes for every model except the majority baseline (section e, per-class table).
- Labels change between catalog versions (section g). The labels used here are from one snapshot.
- Model settings were chosen on the same development folds that are used to report development results. The development scores in section b are therefore not nested estimates.
- `koi_count` may carry information from how multi-planet systems were confirmed.
- Random Forest has a larger train-validation gap than the pruned Decision Tree: 0.1702 vs 0.0468 (section b).
- The centroid columns are vetting diagnostics (section d), so the results describe agreement with the archive disposition, not classification from transit data alone.
- The paired development comparisons use only 5 folds (section c).
