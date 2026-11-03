# Stage 4: Random forest baseline, tuning, and comparison

The feature list is still provisional. This stage uses the same 76 primary features, the 7,381 development rows with a measured transit depth, and the saved `dev_fold` assignments. The median imputer is fit inside each training fold. The test partition is not scored. The experimental significance-statistic configuration is not trained. No result is declared statistically significant.

## Sample and search

- Development rows: 7,381
- Development host stars: 6,357
- Primary features: 76
- Saved split SHA-256: `d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a`
- Seed: 245
- Full listed grid: 216 forests. Evaluated: 25.
- Baseline, pre-specified: `n_estimators` 100, 200, and 400 with `max_depth=None`, `max_features='sqrt'`, `min_samples_leaf=1`, and `class_weight=None`.
- Random phase: 16 configurations drawn with seed 245 from the other 100-tree cells, before any validation score was used.
- Promotion: the 3 random-phase forests with the highest mean validation macro-F1 were repeated at 200 and 400 trees.
- `n_jobs=-1` on 10 CPUs. Fit seconds are the sum of the five training-fold fits and depend on this machine.
- Total training-fold fit time: 91.5 seconds. Trees grown in those fits: 20,500.

Selection uses the maximum mean grouped validation macro-F1. Exact ties break toward the smaller macro-F1 gap, then fewer trees. The tuned score is the score used to choose the forest. It is not a nested estimate on observations held out from the search. The three trees were refit on these same folds so precision and recall use the same metric definition. Their macro-F1 values were checked against the saved Stage 2 and Stage 3 results.

## Tuned forest

n_estimators=100, max_depth=20, max_features=sqrt, min_samples_leaf=2, class_weight=balanced

The search selected a different forest from the 100-tree baseline.

A forest with these hyperparameters was also fit on all development rows for inspection. That fit is in-sample. It was not scored on the test partition.

- In-sample training accuracy: 0.9863
- In-sample training macro-F1: 0.9844
- In-sample fit seconds: 0.3

## Comparison

Validation metrics are means across the five grouped folds, with the sample standard deviation. The macro-F1 gap is mean training macro-F1 minus mean validation macro-F1. Fit seconds are the total of the five training-fold fits.

| Model | Training accuracy | Validation accuracy | Validation macro precision | Validation macro recall | Training macro-F1 | Validation macro-F1 | Macro-F1 gap | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unrestricted tree | 1.0000 ± 0.0000 | 0.7735 ± 0.0089 | 0.7301 ± 0.0098 | 0.7309 ± 0.0074 | 1.0000 ± 0.0000 | 0.7302 ± 0.0086 | 0.2698 | 1.3 |
| Depth-8 tree | 0.8835 ± 0.0067 | 0.8038 ± 0.0052 | 0.7638 ± 0.0055 | 0.7676 ± 0.0114 | 0.8569 ± 0.0075 | 0.7643 ± 0.0075 | 0.0927 | 0.7 |
| Pruned tree | 0.8530 ± 0.0049 | 0.8137 ± 0.0098 | 0.7773 ± 0.0126 | 0.7697 ± 0.0156 | 0.8184 ± 0.0068 | 0.7716 ± 0.0146 | 0.0468 | 1.3 |
| Baseline forest | 1.0000 ± 0.0000 | 0.8488 ± 0.0033 | 0.8202 ± 0.0044 | 0.8020 ± 0.0064 | 1.0000 ± 0.0000 | 0.8084 ± 0.0046 | 0.1916 | 1.2 |
| Tuned forest | 0.9874 ± 0.0013 | 0.8433 ± 0.0105 | 0.8114 ± 0.0133 | 0.8216 ± 0.0133 | 0.9858 ± 0.0016 | 0.8156 ± 0.0130 | 0.1702 | 1.2 |

### Fold-level validation macro-F1

| Fold | Unrestricted tree | Depth-8 tree | Pruned tree | Baseline forest | Tuned forest |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7170 | 0.7768 | 0.7809 | 0.8140 | 0.8243 |
| 1 | 0.7260 | 0.7581 | 0.7548 | 0.8037 | 0.7929 |
| 2 | 0.7361 | 0.7635 | 0.7637 | 0.8125 | 0.8164 |
| 3 | 0.7357 | 0.7587 | 0.7672 | 0.8051 | 0.8216 |
| 4 | 0.7363 | 0.7643 | 0.7915 | 0.8067 | 0.8229 |

Against the pruned tree, the tuned forest has higher validation macro-F1 on 5 of 5 folds, lower on 0, and equal on 0. The paired differences have mean 0.0440, minimum 0.0313, and maximum 0.0544. No hypothesis test was applied. A higher mean is not reported as statistically significant.

Against the baseline forest, the tuned forest has higher validation macro-F1 on 4 of 5 folds, lower on 1, and equal on 0. The paired differences have mean 0.0072, minimum -0.0108, and maximum 0.0165. No hypothesis test was applied. A higher mean is not reported as statistically significant.

## Tuning summary

The five forests with the highest mean grouped validation macro-F1. The full evaluated set is `results/stage4/rf_search_summary.csv`.

| Configuration | Validation macro-F1 | Training macro-F1 | Macro-F1 gap | Fit seconds |
| --- | ---: | ---: | ---: | ---: |
| n_estimators=100, max_depth=20, max_features=sqrt, min_samples_leaf=2, class_weight=balanced | 0.8156 ± 0.0130 | 0.9858 | 0.1702 | 1.2 |
| n_estimators=200, max_depth=20, max_features=sqrt, min_samples_leaf=2, class_weight=balanced | 0.8148 ± 0.0096 | 0.9866 | 0.1718 | 2.1 |
| n_estimators=400, max_depth=20, max_features=sqrt, min_samples_leaf=2, class_weight=balanced | 0.8140 ± 0.0112 | 0.9866 | 0.1726 | 4.1 |
| n_estimators=400, max_depth=None, max_features=sqrt, min_samples_leaf=1, class_weight=None | 0.8131 ± 0.0054 | 1.0000 | 0.1869 | 4.2 |
| n_estimators=100, max_depth=None, max_features=0.5, min_samples_leaf=1, class_weight=balanced | 0.8129 ± 0.0098 | 0.9982 | 0.1853 | 5.0 |

## Out-of-fold class metrics

These precision, recall, and F1 values are computed from pooled out-of-fold predictions. They are not the mean of the five fold scores. The comparison table above remains the selection summary.

| Model | Class | Precision | Recall | F1 | Support |
| --- | --- | ---: | ---: | ---: | ---: |
| Unrestricted tree | CONFIRMED | 0.8181 | 0.8188 | 0.8185 | 2197 |
| Unrestricted tree | CANDIDATE | 0.5146 | 0.5262 | 0.5204 | 1505 |
| Unrestricted tree | FALSE POSITIVE | 0.8559 | 0.8475 | 0.8517 | 3679 |
| Depth-8 tree | CONFIRMED | 0.8113 | 0.8862 | 0.8471 | 2197 |
| Depth-8 tree | CANDIDATE | 0.5962 | 0.5641 | 0.5797 | 1505 |
| Depth-8 tree | FALSE POSITIVE | 0.8819 | 0.8527 | 0.8671 | 3679 |
| Pruned tree | CONFIRMED | 0.8424 | 0.8757 | 0.8587 | 2197 |
| Pruned tree | CANDIDATE | 0.6238 | 0.5475 | 0.5832 | 1505 |
| Pruned tree | FALSE POSITIVE | 0.8628 | 0.8856 | 0.8740 | 3679 |
| Baseline forest | CONFIRMED | 0.8807 | 0.9008 | 0.8906 | 2197 |
| Baseline forest | CANDIDATE | 0.7018 | 0.5754 | 0.6323 | 1505 |
| Baseline forest | FALSE POSITIVE | 0.8769 | 0.9296 | 0.9025 | 3679 |
| Tuned forest | CONFIRMED | 0.8836 | 0.8912 | 0.8874 | 2197 |
| Tuned forest | CANDIDATE | 0.6319 | 0.7003 | 0.6644 | 1505 |
| Tuned forest | FALSE POSITIVE | 0.9185 | 0.8731 | 0.8952 | 3679 |

### Tuned forest confusion matrix

| True \ Predicted | CONFIRMED | CANDIDATE | FALSE POSITIVE |
| --- | ---: | ---: | ---: |
| CONFIRMED | 1958 | 196 | 43 |
| CANDIDATE | 209 | 1054 | 242 |
| FALSE POSITIVE | 49 | 418 | 3212 |

Pooled out-of-fold macro-F1 for the tuned forest: 0.8156.

## Overfitting comparison

Macro-F1 gaps, training minus grouped validation, are 0.2698 for the unrestricted tree, 0.0927 for the depth-8 tree, 0.0468 for the pruned tree, 0.1916 for the baseline forest, and 0.1702 for the tuned forest. The tuned forest's grouped validation macro-F1 is higher than the pruned tree's, and its training-validation macro-F1 gap is not smaller. No hypothesis test was applied.

The tuned forest's mean validation accuracy is 0.8433. The baseline forest's mean validation accuracy is 0.8488. Accuracy is lower for the tuned forest while macro-F1 is higher, which is why the macro-F1 rule selects it. The same hyperparameters at different forest sizes scored 100 trees at 0.8156, 200 trees at 0.8148, 400 trees at 0.8140. A larger forest did not raise validation macro-F1 for this setting. Pooled out-of-fold candidate recall is 0.7003 for the tuned forest and 0.5754 for the baseline forest. Candidate precision is 0.6319 for the tuned forest and 0.7018 for the baseline forest.

## Not done in this stage

The test partition was not scored. Permutation importance was not computed. The significance-statistic configuration was not trained. The 363 incomplete KOIs were not fit. The search did not evaluate every listed forest combination.
