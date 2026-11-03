# Stage 2: unrestricted Decision Tree

The feature list is still provisional. This stage uses the primary leakage-aware features and the development rows that have a measured transit depth. The 363 incomplete KOIs stay in the saved split for a later sensitivity experiment and are not used here. `koi_fwm_stat_sig` is not a training feature. The experimental configuration that adds it is not trained. The test partition is not scored. No forest is trained. No depth is selected as a final model.

## Sample

- Development rows: 7,381
- Development host stars: 6,357
- Primary features: 76
- Saved split SHA-256: `d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a`
- Seed: 245

## Model

Each fold fits a median imputer on its training rows, then a `DecisionTreeClassifier` with `criterion='gini'`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_leaf_nodes=None`, `ccp_alpha=0.0`, `class_weight=None`, and `random_state=245`. The validation curve changes only `max_depth`.

## Grouped cross-validation

Folds are the `dev_fold` values stored in `data/processed/proposed_splits.csv`. They were not recomputed.

| Fold | Train rows | Validation rows | Train accuracy | Validation accuracy | Train macro-F1 | Validation macro-F1 | Depth | Leaves |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5895 | 1486 | 1.0000 | 0.7591 | 1.0000 | 0.7170 | 37 | 585 |
| 1 | 5900 | 1481 | 1.0000 | 0.7718 | 1.0000 | 0.7260 | 30 | 590 |
| 2 | 5900 | 1481 | 1.0000 | 0.7812 | 1.0000 | 0.7361 | 32 | 577 |
| 3 | 5911 | 1470 | 1.0000 | 0.7803 | 1.0000 | 0.7357 | 29 | 595 |
| 4 | 5918 | 1463 | 1.0000 | 0.7751 | 1.0000 | 0.7363 | 28 | 592 |

Mean training accuracy 1.0000 (std 0.0000). Mean grouped validation accuracy 0.7735 (std 0.0089).

Mean training macro-F1 1.0000 (std 0.0000). Mean grouped validation macro-F1 0.7302 (std 0.0086).

Depth by fold: 37, 30, 32, 29, 28 (mean 31.20). Leaves by fold: 585, 590, 577, 595, 592 (mean 587.8).

## In-sample fit on all development rows

This fit is saved so the unrestricted tree can be inspected. Its scores are training scores on the same rows used to grow the tree. They are not a validation estimate, and this model was not applied to the test partition.

- Training accuracy: 1.0000
- Training macro-F1: 1.0000
- Depth: 29
- Leaves: 726

## Validation curve

Values are the mean across the five grouped folds, plus the sample standard deviation. The unrestricted point uses `max_depth=None`.

| max_depth | Train accuracy | Validation accuracy | Train macro-F1 | Validation macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5526 ± 0.0510 | 0.5461 ± 0.0410 | 0.4107 ± 0.0313 | 0.4052 ± 0.0231 |
| 2 | 0.6944 ± 0.0142 | 0.6899 ± 0.0229 | 0.5094 ± 0.0136 | 0.5054 ± 0.0214 |
| 3 | 0.7428 ± 0.0034 | 0.7307 ± 0.0054 | 0.7014 ± 0.0412 | 0.6893 ± 0.0420 |
| 4 | 0.7753 ± 0.0020 | 0.7593 ± 0.0049 | 0.7323 ± 0.0246 | 0.7138 ± 0.0274 |
| 5 | 0.8156 ± 0.0047 | 0.7937 ± 0.0081 | 0.7753 ± 0.0048 | 0.7497 ± 0.0133 |
| 6 | 0.8387 ± 0.0057 | 0.7965 ± 0.0077 | 0.8043 ± 0.0058 | 0.7551 ± 0.0156 |
| 8 | 0.8835 ± 0.0067 | 0.8038 ± 0.0052 | 0.8569 ± 0.0075 | 0.7643 ± 0.0075 |
| 10 | 0.9320 ± 0.0065 | 0.8011 ± 0.0076 | 0.9164 ± 0.0071 | 0.7603 ± 0.0085 |
| 12 | 0.9643 ± 0.0032 | 0.7860 ± 0.0112 | 0.9557 ± 0.0036 | 0.7443 ± 0.0141 |
| 15 | 0.9854 ± 0.0018 | 0.7820 ± 0.0067 | 0.9815 ± 0.0021 | 0.7404 ± 0.0081 |
| 20 | 0.9946 ± 0.0021 | 0.7764 ± 0.0118 | 0.9929 ± 0.0028 | 0.7350 ± 0.0126 |
| 25 | 0.9984 ± 0.0013 | 0.7765 ± 0.0060 | 0.9978 ± 0.0017 | 0.7338 ± 0.0067 |
| 30 | 0.9997 ± 0.0006 | 0.7742 ± 0.0064 | 0.9995 ± 0.0008 | 0.7320 ± 0.0067 |
| 37 | 1.0000 ± 0.0000 | 0.7735 ± 0.0089 | 1.0000 ± 0.0000 | 0.7302 ± 0.0086 |
| unrestricted | 1.0000 ± 0.0000 | 0.7735 ± 0.0089 | 1.0000 ± 0.0000 | 0.7302 ± 0.0086 |

The unrestricted tree has mean training macro-F1 1.0000 and mean grouped validation macro-F1 0.7302. Training minus validation is 0.2698. At max_depth=1, mean training macro-F1 is 0.4107 and mean validation macro-F1 is 0.4052. On this depth grid, the highest mean validation macro-F1 is 0.7643 at max_depth=8, where mean training macro-F1 is 0.8569. Training macro-F1 is higher for the unrestricted tree than for the depth with the highest grouped validation macro-F1, by 0.1431. The unrestricted grouped validation macro-F1 is 0.0341 lower than that depth. The extra depth improves the fit to the training folds and does not improve the grouped validation score. That is the overfitting shown by this development partition.

## Out-of-fold confusion matrix

Each development row is predicted by the unrestricted tree trained on the other four folds.

| True \ Predicted | CONFIRMED | CANDIDATE | FALSE POSITIVE |
| --- | ---: | ---: | ---: |
| CONFIRMED | 1799 | 289 | 109 |
| CANDIDATE | 297 | 792 | 416 |
| FALSE POSITIVE | 103 | 458 | 3118 |

## Not done in this stage

No pruned depth was chosen. The experimental feature set that adds `koi_fwm_stat_sig` was not trained. The 363 incomplete KOIs were not fit. The test partition was not scored. A forest was not trained.
