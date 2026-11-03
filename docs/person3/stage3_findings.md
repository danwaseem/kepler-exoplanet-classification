# Stage 3: Decision Tree tuning and cost-complexity pruning

The feature list is still provisional. This stage uses the same 76 primary features, the 7,381 development rows with a measured transit depth, and the saved `dev_fold` assignments. The imputer is fit inside each training fold. The test partition is not scored. The experimental `koi_fwm_stat_sig` configuration is not trained. A forest is not trained. No configuration is declared statistically significant.

## Sample and search

- Development rows: 7,381
- Development host stars: 6,357
- Primary features: 76
- Saved split SHA-256: `d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a`
- Seed: 245
- Structural grid: max_depth [6, 7, 8, 9, 10, 12], min_samples_split [2, 10, 40], min_samples_leaf [1, 5, 20], ccp_alpha 0 (54 configurations).
- Cost-complexity grid: 22 shared alphas, applied to the unrestricted tree, the depth-8 tree, and the three structural configurations with the highest mean validation macro-F1. Overlapping configurations are fit once. Alpha candidates are summaries of the training-fold pruning paths, not of the validation scores.
- Unique configurations evaluated: 160.

The selection rule is the maximum mean grouped validation macro-F1. If two means are equal, the smaller training-minus-validation macro-F1 gap wins, then the smaller mean leaf count. The reported validation score of the selected configuration is the score used to choose it. It is not a nested estimate on observations held out from the search. Depth 8 was the highest point on the Stage 2 curve for these same folds, so a gain over that tree is not a confirmation on new data.

## Selected configuration

max_depth=unrestricted, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.00111076

Mean depth 11.00 (std 1.41). Mean leaves 44.8 (std 3.4). Macro-F1 gap, training minus grouped validation: 0.0468.

The search selected a different configuration from both the unrestricted tree and the depth-8 tree.

A tree with these hyperparameters was also fit on all development rows so the configuration can be inspected. That fit is in-sample. It was not scored on the test partition.

- In-sample training accuracy: 0.8385
- In-sample training macro-F1: 0.8001
- In-sample depth: 9
- In-sample leaves: 39

## Comparison

The depth-8 tree uses `min_samples_split=2`, `min_samples_leaf=1`, and `ccp_alpha=0`, matching the Stage 2 curve. The unrestricted tree uses the same defaults with no depth limit.

| Model | Train accuracy | Validation accuracy | Train macro-F1 | Validation macro-F1 | Macro-F1 gap | Mean depth | Mean leaves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unrestricted | 1.0000 ± 0.0000 | 0.7735 ± 0.0089 | 1.0000 ± 0.0000 | 0.7302 ± 0.0086 | 0.2698 | 31.20 | 587.8 |
| Depth 8 | 0.8835 ± 0.0067 | 0.8038 ± 0.0052 | 0.8569 ± 0.0075 | 0.7643 ± 0.0075 | 0.0927 | 8.00 | 138.8 |
| Selected | 0.8530 ± 0.0049 | 0.8137 ± 0.0098 | 0.8184 ± 0.0068 | 0.7716 ± 0.0146 | 0.0468 | 11.00 | 44.8 |

### Fold-level validation macro-F1

| Fold | Unrestricted | Depth 8 | Selected | Selected − unrestricted | Selected − depth 8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7170 | 0.7768 | 0.7809 | 0.0639 | 0.0041 |
| 1 | 0.7260 | 0.7581 | 0.7548 | 0.0288 | -0.0033 |
| 2 | 0.7361 | 0.7635 | 0.7637 | 0.0276 | 0.0002 |
| 3 | 0.7357 | 0.7587 | 0.7672 | 0.0315 | 0.0085 |
| 4 | 0.7363 | 0.7643 | 0.7915 | 0.0552 | 0.0272 |

Against the unrestricted tree, the selected configuration has higher validation macro-F1 on 5 of 5 folds, lower on 0, and equal on 0. The paired differences, selected minus unrestricted, have mean 0.0414, minimum 0.0276, and maximum 0.0639. No hypothesis test was applied. A higher mean is not reported as statistically significant.

Against the depth-8 tree, the selected configuration has higher validation macro-F1 on 4 of 5 folds, lower on 1, and equal on 0. The paired differences, selected minus depth-8, have mean 0.0074, minimum -0.0033, and maximum 0.0272. No hypothesis test was applied. A higher mean is not reported as statistically significant.

## Other evaluated configurations

The five highest mean grouped validation macro-F1 values. The full grid is `results/person3/decision_tree/stage3/search_summary.csv`.

| Model | Train accuracy | Validation accuracy | Train macro-F1 | Validation macro-F1 | Macro-F1 gap | Mean depth | Mean leaves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| max_depth=unrestricted, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.00111076 | 0.8530 ± 0.0049 | 0.8137 ± 0.0098 | 0.8184 ± 0.0068 | 0.7716 ± 0.0146 | 0.0468 | 11.00 | 44.8 |
| max_depth=unrestricted, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.00132958 | 0.8428 ± 0.0063 | 0.8140 ± 0.0086 | 0.8056 ± 0.0062 | 0.7713 ± 0.0134 | 0.0342 | 9.80 | 35.0 |
| max_depth=9, min_samples_split=2, min_samples_leaf=20, ccp_alpha=0.00073647 | 0.8528 ± 0.0048 | 0.8108 ± 0.0128 | 0.8193 ± 0.0062 | 0.7691 ± 0.0125 | 0.0501 | 9.00 | 54.2 |
| max_depth=9, min_samples_split=10, min_samples_leaf=20, ccp_alpha=0.00073647 | 0.8528 ± 0.0048 | 0.8108 ± 0.0128 | 0.8193 ± 0.0062 | 0.7691 ± 0.0125 | 0.0501 | 9.00 | 54.2 |
| max_depth=9, min_samples_split=40, min_samples_leaf=20, ccp_alpha=0.00073647 | 0.8528 ± 0.0048 | 0.8108 ± 0.0128 | 0.8193 ± 0.0062 | 0.7691 ± 0.0125 | 0.0501 | 9.00 | 54.2 |

## Cost-complexity pruning

The curve below is the unrestricted tree at each shared alpha. Error bars in the figure are the sample standard deviation across the five folds.

| ccp_alpha | Train accuracy | Validation accuracy | Train macro-F1 | Validation macro-F1 | Mean leaves |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0000 ± 0.0000 | 0.7735 ± 0.0089 | 1.0000 ± 0.0000 | 0.7302 ± 0.0086 | 587.8 |
| 0.000112933 | 0.9999 ± 0.0001 | 0.7736 ± 0.0090 | 0.9999 ± 0.0001 | 0.7303 ± 0.0087 | 586.0 |
| 0.000169492 | 0.9907 ± 0.0028 | 0.7799 ± 0.0107 | 0.9891 ± 0.0033 | 0.7364 ± 0.0121 | 495.2 |
| 0.000209192 | 0.9889 ± 0.0018 | 0.7803 ± 0.0109 | 0.9869 ± 0.0023 | 0.7367 ± 0.0123 | 479.4 |
| 0.000253764 | 0.9824 ± 0.0028 | 0.7807 ± 0.0103 | 0.9791 ± 0.0034 | 0.7370 ± 0.0114 | 437.6 |
| 0.000302955 | 0.9734 ± 0.0022 | 0.7844 ± 0.0098 | 0.9683 ± 0.0026 | 0.7416 ± 0.0103 | 380.4 |
| 0.00032416 | 0.9627 ± 0.0019 | 0.7869 ± 0.0099 | 0.9553 ± 0.0021 | 0.7446 ± 0.0097 | 318.8 |
| 0.000387497 | 0.9548 ± 0.0019 | 0.7926 ± 0.0073 | 0.9458 ± 0.0022 | 0.7492 ± 0.0092 | 274.4 |
| 0.00040652 | 0.9530 ± 0.0026 | 0.7930 ± 0.0077 | 0.9435 ± 0.0030 | 0.7498 ± 0.0092 | 265.6 |
| 0.0005035 | 0.9370 ± 0.0029 | 0.7983 ± 0.0088 | 0.9236 ± 0.0036 | 0.7537 ± 0.0120 | 208.2 |
| 0.000592331 | 0.9184 ± 0.0015 | 0.8020 ± 0.0100 | 0.9008 ± 0.0019 | 0.7591 ± 0.0135 | 155.8 |
| 0.000717781 | 0.8937 ± 0.0037 | 0.8058 ± 0.0112 | 0.8703 ± 0.0042 | 0.7635 ± 0.0125 | 103.6 |
| 0.00073647 | 0.8907 ± 0.0047 | 0.8063 ± 0.0107 | 0.8668 ± 0.0053 | 0.7649 ± 0.0111 | 98.8 |
| 0.00111076 | 0.8530 ± 0.0049 | 0.8137 ± 0.0098 | 0.8184 ± 0.0068 | 0.7716 ± 0.0146 | 44.8 |
| 0.00132958 | 0.8428 ± 0.0063 | 0.8140 ± 0.0086 | 0.8056 ± 0.0062 | 0.7713 ± 0.0134 | 35.0 |
| 0.00246286 | 0.8227 ± 0.0028 | 0.8063 ± 0.0093 | 0.7804 ± 0.0064 | 0.7592 ± 0.0151 | 18.4 |
| 0.00456208 | 0.8146 ± 0.0040 | 0.8021 ± 0.0070 | 0.7747 ± 0.0052 | 0.7605 ± 0.0111 | 13.0 |
| 0.00845057 | 0.8062 ± 0.0080 | 0.7964 ± 0.0097 | 0.7667 ± 0.0101 | 0.7566 ± 0.0147 | 9.2 |
| 0.0156534 | 0.7756 ± 0.0108 | 0.7655 ± 0.0138 | 0.7311 ± 0.0292 | 0.7209 ± 0.0307 | 6.8 |
| 0.0289957 | 0.7381 ± 0.0082 | 0.7258 ± 0.0110 | 0.6819 ± 0.0831 | 0.6707 ± 0.0836 | 5.0 |
| 0.0537102 | 0.6893 ± 0.0255 | 0.6839 ± 0.0364 | 0.5057 ± 0.0218 | 0.5011 ± 0.0309 | 2.8 |
| 0.0994902 | 0.5684 ± 0.0977 | 0.5614 ± 0.0916 | 0.3298 ± 0.1489 | 0.3243 ± 0.1426 | 1.6 |

At ccp_alpha=0 the unrestricted tree has mean training macro-F1 1.0000 and mean validation macro-F1 0.7302. On this shared alpha grid, the highest mean validation macro-F1 is 0.7716 at ccp_alpha=0.00111076, where mean training macro-F1 is 0.8184 and the mean leaf count is 44.8. At the largest evaluated alpha (0.0994902), mean training macro-F1 is 0.3298 and mean validation macro-F1 is 0.3243. Raising ccp_alpha from 0 to 0.0994902 lowers mean training macro-F1 by 0.6702. ccp_alpha=0.00111076 changes grouped validation macro-F1 by 0.0414 relative to the unpruned unrestricted tree.

## Tradeoff

The selected tree's mean validation macro-F1 is 0.7716, compared with 0.7643 for the depth-8 tree and 0.7302 for the unrestricted tree. Its macro-F1 gap is 0.0468, compared with 0.0927 at depth 8 and 0.2698 when the tree is unrestricted. Mean leaves are 44.8 for the selected tree, 138.8 at depth 8, and 587.8 for the unrestricted tree. Relative to depth 8, the selected tree has a higher grouped validation macro-F1 and a smaller training-validation gap on these folds. The next configuration, the same unrestricted tree at ccp_alpha=0.00132958, has mean validation macro-F1 0.7713, mean validation accuracy 0.8140, a macro-F1 gap of 0.0342, and 35.0 mean leaves. The selected alpha leads that neighbor by 0.0003 macro-F1, while the neighbor has slightly higher validation accuracy and a smaller gap. The depth-9 trees with min_samples_leaf=20 and min_samples_split of 2, 10, or 40 are tied with each other, because a minimum leaf size of 20 already blocks the splits those three settings would otherwise distinguish.

## Not done in this stage

The test partition was not scored. The `koi_fwm_stat_sig` configuration was not trained. The 363 incomplete KOIs were not fit. A forest was not trained. The search did not cross every structural combination with every alpha. No significance test was used to choose the configuration.
