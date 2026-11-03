# Stage 5B: centroid-offset sensitivity

This is a limited sensitivity check on the development partition. It does not replace the 76-feature primary set or the selected tree and forest. Configuration C adds `koi_fwm_stat_sig` as a diagnostic feature. That column is a potentially vetting-derived summary, not an accepted production feature. The test partition is not scored.

## Configurations

- A: the existing primary set, 76 columns.
- B: A without 10 provisional centroid-offset measurements and their 10 uncertainty columns, leaving 56 columns.
- C: A plus `koi_fwm_stat_sig`. Diagnostic only. The column page says this percent indicates whether there is a statistically significant flux-weighted offset between in-transit and out-of-transit images, and that 100% means there is no offset and there is confidence that the transit is on the target star.
- Development rows: 7,381. Host stars: 6,357.
- Saved split SHA-256: `d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a`
- Missing fraction of `koi_fwm_stat_sig` on development rows: 0.1051. The median imputer is fit inside each training fold.
- Hyperparameters are the Stage 3 pruned tree and the Stage 4 tuned forest. Configuration A reproduces their saved grouped validation macro-F1.

## Grouped validation

Means and standard deviations are across the five saved development folds.

| Model | Config | Features | Validation accuracy | Validation macro-F1 | Confirmed recall | Candidate recall | False Positive recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pruned tree | A | 76 | 0.8137 ± 0.0098 | 0.7716 ± 0.0146 | 0.8757 ± 0.0203 | 0.5476 ± 0.0605 | 0.8856 ± 0.0198 |
| Pruned tree | B | 56 | 0.7705 ± 0.0113 | 0.7203 ± 0.0198 | 0.8334 ± 0.0224 | 0.4623 ± 0.0912 | 0.8591 ± 0.0373 |
| Pruned tree | C (diagnostic) | 77 | 0.8121 ± 0.0076 | 0.7699 ± 0.0126 | 0.8712 ± 0.0182 | 0.5487 ± 0.0643 | 0.8847 ± 0.0127 |
| Tuned forest | A | 76 | 0.8433 ± 0.0105 | 0.8156 ± 0.0130 | 0.8912 ± 0.0136 | 0.7005 ± 0.0355 | 0.8731 ± 0.0121 |
| Tuned forest | B | 56 | 0.7896 ± 0.0056 | 0.7634 ± 0.0065 | 0.8607 ± 0.0154 | 0.6561 ± 0.0260 | 0.8019 ± 0.0092 |
| Tuned forest | C (diagnostic) | 77 | 0.8408 ± 0.0066 | 0.8133 ± 0.0091 | 0.8917 ± 0.0134 | 0.7004 ± 0.0266 | 0.8679 ± 0.0066 |

## Paired differences relative to A

Each difference is the named configuration minus A on the same validation fold.

| Model | Contrast | Metric | Mean difference | Std | Min | Max | Higher | Lower |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pruned tree | B-A | validation_macro_f1 | -0.0513 | 0.0176 | -0.0754 | -0.0307 | 0 | 5 |
| Pruned tree | B-A | validation_accuracy | -0.0433 | 0.0138 | -0.0643 | -0.0284 | 0 | 5 |
| Pruned tree | B-A | validation_recall_candidate | -0.0854 | 0.0721 | -0.2079 | -0.0233 | 0 | 5 |
| Pruned tree | B-A | validation_recall_confirmed | -0.0423 | 0.0253 | -0.0752 | -0.0205 | 0 | 5 |
| Pruned tree | B-A | validation_recall_false_positive | -0.0265 | 0.0416 | -0.0989 | 0.0014 | 1 | 4 |
| Pruned tree | C-A | validation_macro_f1 | -0.0017 | 0.0054 | -0.0097 | 0.0052 | 1 | 3 |
| Pruned tree | C-A | validation_accuracy | -0.0016 | 0.0042 | -0.0068 | 0.0027 | 2 | 2 |
| Pruned tree | C-A | validation_recall_candidate | 0.0011 | 0.0508 | -0.0852 | 0.0441 | 3 | 1 |
| Pruned tree | C-A | validation_recall_confirmed | -0.0046 | 0.0101 | -0.0182 | 0.0068 | 1 | 2 |
| Pruned tree | C-A | validation_recall_false_positive | -0.0009 | 0.0276 | -0.0288 | 0.0445 | 1 | 3 |
| Tuned forest | B-A | validation_macro_f1 | -0.0522 | 0.0138 | -0.0675 | -0.0318 | 0 | 5 |
| Tuned forest | B-A | validation_accuracy | -0.0537 | 0.0114 | -0.0659 | -0.0371 | 0 | 5 |
| Tuned forest | B-A | validation_recall_candidate | -0.0444 | 0.0325 | -0.0852 | 0.0033 | 1 | 4 |
| Tuned forest | B-A | validation_recall_confirmed | -0.0305 | 0.0070 | -0.0388 | -0.0228 | 0 | 5 |
| Tuned forest | B-A | validation_recall_false_positive | -0.0712 | 0.0112 | -0.0823 | -0.0541 | 0 | 5 |
| Tuned forest | C-A | validation_macro_f1 | -0.0023 | 0.0050 | -0.0072 | 0.0055 | 1 | 4 |
| Tuned forest | C-A | validation_accuracy | -0.0024 | 0.0044 | -0.0068 | 0.0047 | 1 | 4 |
| Tuned forest | C-A | validation_recall_candidate | -0.0000 | 0.0139 | -0.0133 | 0.0165 | 2 | 3 |
| Tuned forest | C-A | validation_recall_confirmed | 0.0005 | 0.0037 | -0.0046 | 0.0046 | 3 | 2 |
| Tuned forest | C-A | validation_recall_false_positive | -0.0052 | 0.0066 | -0.0124 | 0.0054 | 1 | 4 |

- Pruned tree, B-A macro-F1: Mean paired difference -0.0513 (std 0.0176), from -0.0754 to -0.0307. The alternative is higher on 0 folds, lower on 5 folds, and equal on 0 folds. This count is not a hypothesis test.
- Pruned tree, C-A macro-F1: Mean paired difference -0.0017 (std 0.0054), from -0.0097 to 0.0052. The alternative is higher on 1 fold, lower on 3 folds, and equal on 1 fold. This count is not a hypothesis test.
- Tuned forest, B-A macro-F1: Mean paired difference -0.0522 (std 0.0138), from -0.0675 to -0.0318. The alternative is higher on 0 folds, lower on 5 folds, and equal on 0 folds. This count is not a hypothesis test.
- Tuned forest, C-A macro-F1: Mean paired difference -0.0023 (std 0.0050), from -0.0072 to 0.0055. The alternative is higher on 1 fold, lower on 4 folds, and equal on 0 folds. This count is not a hypothesis test.
- Pruned tree, B-A Candidate recall: Mean paired difference -0.0854 (std 0.0721), from -0.2079 to -0.0233. The alternative is higher on 0 folds, lower on 5 folds, and equal on 0 folds. This count is not a hypothesis test.
- Pruned tree, C-A Candidate recall: Mean paired difference 0.0011 (std 0.0508), from -0.0852 to 0.0441. The alternative is higher on 3 folds, lower on 1 fold, and equal on 1 fold. This count is not a hypothesis test.
- Tuned forest, B-A Candidate recall: Mean paired difference -0.0444 (std 0.0325), from -0.0852 to 0.0033. The alternative is higher on 1 fold, lower on 4 folds, and equal on 0 folds. This count is not a hypothesis test.
- Tuned forest, C-A Candidate recall: Mean paired difference -0.0000 (std 0.0139), from -0.0133 to 0.0165. The alternative is higher on 2 folds, lower on 3 folds, and equal on 0 folds. This count is not a hypothesis test.

## Candidate rows

The question is whether removing the centroid offsets changes Candidate recall or moves Candidate rows into Confirmed or False Positive. Counts are pooled out-of-fold predictions on the same 7,381 development rows.

For the Pruned tree, pooled development out-of-fold Candidate recall is 0.5475 under A and 0.4618 under B. Candidate rows predicted Confirmed change from 260 to 257. Candidate rows predicted False Positive change from 421 to 553.
For the Pruned tree, pooled development out-of-fold Candidate recall is 0.5475 under A and 0.5482 under diagnostic C. Candidate rows predicted Confirmed change from 260 to 258. Candidate rows predicted False Positive change from 421 to 422.
For the Tuned forest, pooled development out-of-fold Candidate recall is 0.7003 under A and 0.6558 under B. Candidate rows predicted Confirmed change from 209 to 209. Candidate rows predicted False Positive change from 242 to 309.
For the Tuned forest, pooled development out-of-fold Candidate recall is 0.7003 under A and 0.7003 under diagnostic C. Candidate rows predicted Confirmed change from 209 to 206. Candidate rows predicted False Positive change from 242 to 245.

Removing the centroid block does not increase the number of Candidate rows predicted as Confirmed. Those pooled counts go from 260 to 257 for the pruned tree and stay at 209 for the tuned forest. The Candidate recall loss appears as False Positive predictions: they go from 421 to 553 for the tree and go from 242 to 309 for the forest. The change is broader than the Candidate row. For the tuned forest, False Positive rows predicted Confirmed go from 49 to 164, False Positive rows predicted Candidate go from 418 to 565, and Confirmed rows predicted False Positive go from 43 to 89.

Configuration C remains a diagnostic experiment. A higher or lower score with `koi_fwm_stat_sig` included does not make that column part of the primary feature set. On these folds its mean validation macro-F1 is slightly below configuration A for both models, and the fold differences change sign.

## Limitations

The same development folds were used to select the hyperparameters, so these scores are not a fresh confirmation of that selection. Fold counts are not a hypothesis test. Correlated centroid columns can share a signal, so removing the whole block is a different question from permuting one column. The primary models and the primary feature configuration were not changed.

## Not done in this stage

The test partition was not scored. No new primary model or feature list was selected.
