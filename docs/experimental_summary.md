# Experimental summary, Stages 1–5B

This summary collects the saved development-partition results for Danish Waseem's Decision Tree and Random Forest experiments. It does not add a new model, a new split, or a test-set score. Metrics are copied from the stage findings and result files. No paper or DOI was added here. The data DOI already recorded in Stage 1 is 10.26133/NEA4.

## Decision for the primary experiment

The primary experiment keeps the existing 76-feature leakage-aware configuration. The 56-feature configuration that drops provisional centroid offsets remains a documented sensitivity analysis. The 77-feature configuration that adds `koi_fwm_stat_sig` remains diagnostic only. The selected hyperparameters, feature lists, and saved splits were not changed after Stage 5B. The test partition has not been scored.

The team proposal file is still marked not team-final. That status was not edited.

## Data, split, and leakage

The table is the NASA Exoplanet Archive Kepler Objects of Interest cumulative table, filename version date 2026-10-06, local file `data/raw/koi_cumulative_2026-10-06.csv`. SHA-256 `a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab`, 11,843,781 bytes, 9,564 rows and 153 columns. There are 0 duplicate rows and 0 duplicate `kepoi_name` values. There are 8,214 host stars (`kepid`), 938 stars with more than one KOI (2,288 KOIs), and 327 stars with mixed dispositions.

Full-table dispositions: CONFIRMED 2,748 (28.7%), CANDIDATE 1,977 (20.7%), FALSE POSITIVE 4,839 (50.6%). FALSE POSITIVE is the largest class. That share is a base-rate context, not a trained baseline.

The column page was retrieved on 2026-10-31. The saved page says it was last updated on 2021-02-11.

The target is `koi_disposition`. The grouping key is `kepid`. Seed 245. The holdout is `StratifiedGroupKFold` with 5 folds, shuffle on, and test equal to fold 0. Development cross-validation is a separate 5-fold grouped split on the development stars, stored as `dev_fold`. Saved split SHA-256: `d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a`. Host-star overlap between development and test is zero.

| Slice | Rows | Host stars | CONFIRMED | CANDIDATE | FALSE POSITIVE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development, all assigned rows | 7,653 | 6,573 | 28.7% | 20.7% | 50.6% |
| Test, all assigned rows | 1,911 | 1,641 | 28.7% | 20.7% | 50.6% |
| Development, measured transit depth | 7,381 | 6,357 | 29.8% | 20.4% | 49.8% |
| Test, measured transit depth | 1,820 | 1,570 | 30.2% | 20.2% | 49.6% |

The primary matrix uses the 7,381 development rows with a measured `koi_depth`. Supports in that matrix are CONFIRMED 2,197, CANDIDATE 1,505, and FALSE POSITIVE 3,679. The 363 rows with missing `koi_depth` stay in the split file for a later sensitivity experiment. They are not deleted and they are not median-imputed. Six rows with fit type none still have a depth and stay in the primary matrix.

Excluded from the scientific features: `koi_pdisposition`, `koi_score`, the four `koi_fpflag_*` columns, `kepler_name`, and `koi_comment`. `kepid` is the grouping key only. `koi_fwm_stat_sig` is excluded from the primary 76 columns and kept only in the diagnostic configuration. Centroid-offset measurements and their uncertainty columns are inside the 76 and are marked provisional. `koi_count` equals the number of KOI rows on a star, including false positives, on every host star in this file. The column page calls it the number of planet candidates.

Preprocessing for these models is a median imputer inside the training pipeline. Trees and forests are not standardized. The imputer is fit on the training fold only. Residual gaps on depth-measured rows are the cells that get imputed.

## Selected models

Both models use Gini splits, `random_state=245`, and the median imputer.

- Pruned tree: `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `ccp_alpha=0.0011107571059714625`, `class_weight=None`.
- Tuned forest: `n_estimators=100`, `max_depth=20`, `max_features='sqrt'`, `min_samples_leaf=2`, `min_samples_split=2`, `class_weight='balanced'`, `bootstrap=True`.

Selection maximized mean grouped validation macro-F1. Exact ties would break toward the smaller macro-F1 gap, then fewer leaves or fewer trees. The selected score is the score used to choose the model. It is not a nested estimate. No hypothesis test was applied.

## Stage 2: unrestricted tree

The unrestricted tree memorizes each training fold: training accuracy and macro-F1 are 1.0000. Mean grouped validation accuracy is 0.7735 ± 0.0089. Mean grouped validation macro-F1 is 0.7302 ± 0.0086. The macro-F1 gap is 0.2698. Fold depths are 37, 30, 32, 29, and 28. Fold leaf counts are 585, 590, 577, 595, and 592.

On the depth grid, validation macro-F1 is highest at `max_depth=8`: 0.7643 ± 0.0075, with training macro-F1 0.8569 ± 0.0075. The unrestricted validation macro-F1 is 0.0341 lower. At `max_depth=1`, training macro-F1 is 0.4107 and validation macro-F1 is 0.4052. Depths 7 and 9 were not on that grid. No depth was selected as the final tree in Stage 2.

## Stage 3: pruning

The search had 54 structural configurations and 22 shared cost-complexity alphas, 160 unique configurations after duplicates were removed. Alphas came from the training-fold pruning paths, not from the validation scores.

| Model | Train accuracy | Validation accuracy | Train macro-F1 | Validation macro-F1 | Gap | Mean depth | Mean leaves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unrestricted | 1.0000 ± 0.0000 | 0.7735 ± 0.0089 | 1.0000 ± 0.0000 | 0.7302 ± 0.0086 | 0.2698 | 31.20 | 587.8 |
| Depth 8 | 0.8835 ± 0.0067 | 0.8038 ± 0.0052 | 0.8569 ± 0.0075 | 0.7643 ± 0.0075 | 0.0927 | 8.00 | 138.8 |
| Pruned tree | 0.8530 ± 0.0049 | 0.8137 ± 0.0098 | 0.8184 ± 0.0068 | 0.7716 ± 0.0146 | 0.0468 | 11.00 | 44.8 |

The pruned tree is higher than the unrestricted tree on 5 of 5 folds. Paired differences have mean 0.0414, minimum 0.0276, and maximum 0.0639. It is higher than the depth-8 tree on 4 of 5 folds and lower on fold 1 by 0.0033. The mean paired difference versus depth 8 is 0.0074, minimum −0.0033, maximum 0.0272. Depth 8 was the peak of the Stage 2 curve on these same folds, so that comparison is not a new sample.

The next alpha, 0.00132958, has validation macro-F1 0.7713, validation accuracy 0.8140, gap 0.0342, and 35.0 mean leaves. The selected alpha leads it by about 0.0003 macro-F1. An in-sample fit on all development rows has depth 9 and 39 leaves. That fit was not used to score validation folds.

## Stage 4: forest comparison

The listed forest grid has 216 cells. The search evaluated 25. The baseline architecture was pre-specified at 100, 200, and 400 trees. Sixteen other 100-tree cells were drawn with seed 245 before validation scores were seen. The top three of those sixteen were repeated at 200 and 400 trees.

| Model | Train accuracy | Validation accuracy | Validation macro precision | Validation macro recall | Train macro-F1 | Validation macro-F1 | Gap | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unrestricted tree | 1.0000 ± 0.0000 | 0.7735 ± 0.0089 | 0.7301 ± 0.0098 | 0.7309 ± 0.0074 | 1.0000 ± 0.0000 | 0.7302 ± 0.0086 | 0.2698 | 1.3 |
| Depth-8 tree | 0.8835 ± 0.0067 | 0.8038 ± 0.0052 | 0.7638 ± 0.0055 | 0.7676 ± 0.0114 | 0.8569 ± 0.0075 | 0.7643 ± 0.0075 | 0.0927 | 0.7 |
| Pruned tree | 0.8530 ± 0.0049 | 0.8137 ± 0.0098 | 0.7773 ± 0.0126 | 0.7697 ± 0.0156 | 0.8184 ± 0.0068 | 0.7716 ± 0.0146 | 0.0468 | 1.3 |
| Baseline forest | 1.0000 ± 0.0000 | 0.8488 ± 0.0033 | 0.8202 ± 0.0044 | 0.8020 ± 0.0064 | 1.0000 ± 0.0000 | 0.8084 ± 0.0046 | 0.1916 | 1.2 |
| Tuned forest | 0.9874 ± 0.0013 | 0.8433 ± 0.0105 | 0.8114 ± 0.0133 | 0.8216 ± 0.0133 | 0.9858 ± 0.0016 | 0.8156 ± 0.0130 | 0.1702 | 1.2 |

Fold validation macro-F1:

| Fold | Unrestricted tree | Depth-8 tree | Pruned tree | Baseline forest | Tuned forest |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7170 | 0.7768 | 0.7809 | 0.8140 | 0.8243 |
| 1 | 0.7260 | 0.7581 | 0.7548 | 0.8037 | 0.7929 |
| 2 | 0.7361 | 0.7635 | 0.7637 | 0.8125 | 0.8164 |
| 3 | 0.7357 | 0.7587 | 0.7672 | 0.8051 | 0.8216 |
| 4 | 0.7363 | 0.7643 | 0.7915 | 0.8067 | 0.8229 |

The tuned forest is higher than the pruned tree on 5 of 5 folds. Paired differences have mean 0.0440, minimum 0.0313, and maximum 0.0544. Its gap, 0.1702, is larger than the pruned tree's gap, 0.0468. The forest raises validation macro-F1 and does not shrink the training-validation gap relative to the pruned tree. Against the unrestricted tree, both the validation score and the gap move in the direction the research question asked about. Against the baseline forest, the tuned forest is higher on 4 of 5 folds and lower on fold 1 by 0.0108. The mean paired difference is 0.0072.

The same tuned architecture at 200 trees has validation macro-F1 0.8148, and at 400 trees 0.8140. More trees did not raise that score. The baseline forest at 400 trees has validation macro-F1 0.8131. Fit times are machine-dependent. The five-fold fit totals above are about 1.2 to 1.3 seconds for the selected models.

`class_weight='balanced'` lowers validation accuracy relative to the unweighted baseline forest, 0.8433 versus 0.8488, and raises macro-F1. Pooled out-of-fold Candidate recall is 0.7003 for the tuned forest and 0.5754 for the baseline forest. Candidate precision is 0.6319 and 0.7018.

Pooled out-of-fold class metrics for the two selected models:

| Model | Class | Precision | Recall | F1 | Support |
| --- | --- | ---: | ---: | ---: | ---: |
| Pruned tree | CONFIRMED | 0.8424 | 0.8757 | 0.8587 | 2197 |
| Pruned tree | CANDIDATE | 0.6238 | 0.5475 | 0.5832 | 1505 |
| Pruned tree | FALSE POSITIVE | 0.8628 | 0.8856 | 0.8740 | 3679 |
| Tuned forest | CONFIRMED | 0.8836 | 0.8912 | 0.8874 | 2197 |
| Tuned forest | CANDIDATE | 0.6319 | 0.7003 | 0.6644 | 1505 |
| Tuned forest | FALSE POSITIVE | 0.9185 | 0.8731 | 0.8952 | 3679 |

Tuned forest confusion matrix, rows true and columns predicted, order CONFIRMED, CANDIDATE, FALSE POSITIVE: 1958, 196, 43; 209, 1054, 242; 49, 418, 3212. Pooled out-of-fold macro-F1 is 0.8156.

## Stage 5: feature importance

Permutation importance is the drop in validation macro-F1 when one raw feature is shuffled, 10 repeats, aggregated across the five validation folds. Each model and its imputer were fit on the training fold only. Impurity importance is the training-fold split improvement and is descriptive. Importance did not change the selected models.

Stable forest permutation features, mean drop and across-fold standard deviation:

| Rank | Column | Mean drop | Std | Fold ranks | Provisional centroid |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `koi_model_snr` | 0.0183 | 0.0050 | 1–2 | No |
| 2 | `koi_dikco_msky` | 0.0150 | 0.0028 | 1–2 | Yes |
| 3 | `koi_dicco_msky` | 0.0088 | 0.0026 | 3 on every fold | Yes |

The column page defines `koi_model_snr` as transit depth divided by the mean uncertainty in the flux during the transits. `koi_dikco_msky` is the sky-plane angle between the difference-image PRF centroid and the Kepler Input Catalog position. `koi_dicco_msky` is the sky-plane angle between the out-of-transit PRF centroid and the difference-image centroid. Those two offsets have Spearman correlation 0.8603 on the development rows.

Mean ranks from 4 downward are about 0.002, with across-fold standard deviations as large as the means. The mean pairwise Spearman correlation of the full 76-feature forest importance vectors is 0.1780. The pre-specified stability rule for the top 10 is not met. Only the first three features hold their ranks.

The pruned tree's largest permutation drop is `koi_prad` at 0.1382. The same shuffle drops the forest by 0.0024. Six of each model's ten leading features are shared. The Spearman correlation between the two mean importance vectors is 0.5762. Impurity importance has rank correlation 0.6325 with forest permutation importance and places `koi_ror` and `koi_prad` at impurity ranks 4 and 5.

The pre-specified centroid-dominance rule was not met: signal-to-noise is first, 2 of the forest's top 10 are centroid offsets, and those offsets hold 0.4497 of the positive mean permutation importance. Their stable second and third place is still a methodological concern.

Other development-row Spearman associations at or above 0.7 include `koi_period` with `koi_sma` (0.9966), `koi_teq` with `koi_insol` (1.0000), `koi_slogg` with `koi_srad` (−0.9796), `koi_depth` with `koi_ror` (0.9379), and `koi_prad` with `koi_ror` (0.8840). Permutation importance does not establish causation. Correlated columns can share a score drop.

## Stage 5B: centroid sensitivity

Configuration A is the 76-feature primary set and reproduces the Stage 3 and Stage 4 validation macro-F1. Configuration B drops 10 centroid-offset measurements and their 10 uncertainty columns. Configuration C adds `koi_fwm_stat_sig` to A. That column is missing on 10.51% of development rows. The column page says 100% means there is no significant flux-weighted offset and there is confidence that the transit is on the target star.

| Model | Config | Features | Validation accuracy | Validation macro-F1 | Candidate recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pruned tree | A, primary | 76 | 0.8137 ± 0.0098 | 0.7716 ± 0.0146 | 0.5476 ± 0.0605 |
| Pruned tree | B, sensitivity | 56 | 0.7705 ± 0.0113 | 0.7203 ± 0.0198 | 0.4623 ± 0.0912 |
| Pruned tree | C, diagnostic | 77 | 0.8121 ± 0.0076 | 0.7699 ± 0.0126 | 0.5487 ± 0.0643 |
| Tuned forest | A, primary | 76 | 0.8433 ± 0.0105 | 0.8156 ± 0.0130 | 0.7005 ± 0.0355 |
| Tuned forest | B, sensitivity | 56 | 0.7896 ± 0.0056 | 0.7634 ± 0.0065 | 0.6561 ± 0.0260 |
| Tuned forest | C, diagnostic | 77 | 0.8408 ± 0.0066 | 0.8133 ± 0.0091 | 0.7004 ± 0.0266 |

B minus A validation macro-F1 is −0.0513 for the tree and −0.0522 for the forest, lower on all five folds. Pooled Candidate recall goes from 0.5475 to 0.4618 for the tree and from 0.7003 to 0.6558 for the forest. Candidate rows predicted Confirmed go from 260 to 257 for the tree and stay at 209 for the forest. Candidate rows predicted False Positive go from 421 to 553 for the tree and from 242 to 309 for the forest. For the forest, False Positive rows predicted Confirmed go from 49 to 164, False Positive rows predicted Candidate go from 418 to 565, and Confirmed rows predicted False Positive go from 43 to 89.

C minus A validation macro-F1 is −0.0017 for the tree and −0.0023 for the forest. The fold differences change sign. Candidate recall is essentially unchanged. C was not adopted.

## Limitations

- The reported validation scores for the selected tree and forest are the scores used to choose them. They are not nested cross-validation estimates. The Stage 3 comparison with depth 8 uses the same folds that identified depth 8 in Stage 2.
- Fold counts and mean differences are not hypothesis tests.
- The 76-feature list is retained for this primary experiment and is still a team proposal.
- Centroid offsets remain vetting measurements. Removing them lowers development validation macro-F1. Keeping them means the model can use the centroid analysis. The primary set was not switched to the 56-feature sensitivity configuration.
- Permutation importance depends on the other columns that remain available. It is not a causal ranking. Impurity importance answers a different question.
- Only three forest permutation features are stable across folds.
- `koi_fwm_stat_sig` is diagnostic. It did not improve grouped validation macro-F1.
- The 363 rows without a transit depth have not been used in a sensitivity fit.
- The forest search left 191 of 216 listed cells unevaluated.
- Fit seconds depend on the machine.
- The test partition has not been scored.

## Remaining tasks

- Final test evaluation has not started. It should use the retained 76-feature set and the selected hyperparameters, after a separate decision to open the test partition.
- The 363 incomplete KOIs remain reserved for a sensitivity experiment.
- The literature review of at least 13 papers, in the student's own words with APA citations, has not been written. The intermediate report is due Tuesday, November 3, 2026, at 4:59 PM.
- The team has not marked the feature list and split as final.
- The generative-AI disclosure still needs this log, or the portion the course requires, in the student's own words.

## Figures for the intermediate report

Each caption describes the saved figure. None of these figures uses the test partition.

| File | Caption |
| --- | --- |
| `figures/eda/class_balance.png` | Disposition counts in the cumulative table. |
| `figures/eda/missingness.png` | Missingness among columns considered for the learning matrix. |
| `figures/eda/unfitted_block_by_disposition.png` | Disposition mix for rows with and without a measured transit depth. |
| `figures/eda/feature_distributions_by_class.png` | Selected measurement distributions by archive disposition. |
| `figures/eda/spearman_correlation.png` | Spearman correlations among a subset of development measurements. |
| `figures/stage2/validation_curve.png` | Training and grouped validation macro-F1 against Decision Tree `max_depth`, including the unrestricted tree. |
| `figures/stage2/oof_confusion_matrix.png` | Unrestricted tree, pooled development out-of-fold confusion matrix. |
| `figures/stage3/ccp_alpha_curve.png` | Training and grouped validation performance against `ccp_alpha` for the unrestricted tree. |
| `figures/stage3/model_comparison.png` | Training and grouped validation macro-F1 for the unrestricted tree, the depth-8 tree, and the pruned tree. |
| `figures/stage4/model_comparison.png` | Training and grouped validation macro-F1 for the three trees, the baseline forest, and the tuned forest. |
| `figures/stage4/oof_confusion_matrix.png` | Tuned forest, pooled development out-of-fold confusion matrix. |
| `figures/stage5/forest_permutation_importance.png` | Fifteen largest tuned-forest permutation importances. Error bars are the standard deviation across the five validation folds. Pink bars are provisional centroid offsets. |
| `figures/stage5/forest_impurity_importance.png` | Fifteen largest tuned-forest impurity importances, averaged over the training folds. |
| `figures/stage5/forest_rank_by_fold.png` | Fold-by-fold permutation rank of the ten features with the highest mean forest importance. |
| `figures/stage5/tree_vs_forest_permutation.png` | Mean permutation importance for features in either model's top ten. An asterisk marks a provisional centroid offset. |
| `figures/stage5/top_feature_correlation.png` | Spearman correlation on development rows among the leading forest features and strongly associated partners. |
| `figures/stage5b/validation_macro_f1.png` | Mean grouped validation macro-F1 for configurations A, B, and C. Error bars are the standard deviation across folds. |
| `figures/stage5b/candidate_recall.png` | Mean grouped validation Candidate recall for configurations A, B, and C. |
| `figures/stage5b/paired_macro_f1_differences.png` | Fold-level validation macro-F1 minus configuration A. Configuration C is diagnostic. |
| `figures/stage5b/tree_confusion_A_vs_B.png` | Pruned tree, pooled development out-of-fold confusion matrices for the 76-feature set and the 56-feature sensitivity set. |
| `figures/stage5b/forest_confusion_A_vs_B.png` | Tuned forest, pooled development out-of-fold confusion matrices for the 76-feature set and the 56-feature sensitivity set. |
| `figures/stage5b/forest_confusion_A_vs_C_diagnostic.png` | Tuned forest, pooled development out-of-fold confusion matrices for the primary set and the diagnostic set that adds `koi_fwm_stat_sig`. |

## Tables for the intermediate report

The stage findings files repeat these tables with the same numbers. The files below are the saved sources.

| File | Caption |
| --- | --- |
| `results/stage1/dataset_record.json` | Checksum, shape, source, and split settings for the cumulative table. |
| `results/stage1/split_balance.json` | Row counts, host-star counts, and disposition shares by partition. |
| `results/stage1/column_audit.csv` | Column-by-column include or exclude decision, with the column-page reason. |
| `data/processed/proposed_leakage_aware_features.csv` | The retained 76-feature primary list. |
| `data/processed/proposed_splits.csv` | Saved `kepid` split and development fold for every KOI. |
| `results/stage2/validation_curve.csv` | Depth-grid training and validation accuracy and macro-F1. |
| `results/stage2/fold_metrics.csv` | Unrestricted-tree metrics on each development fold. |
| `results/stage2/oof_confusion_matrix.csv` | Unrestricted-tree pooled out-of-fold confusion counts. |
| `results/stage3/comparison_folds.csv` | Fold metrics for the unrestricted tree, the depth-8 tree, and the pruned tree. |
| `results/stage3/ccp_curve.csv` | Shared `ccp_alpha` grid for the unrestricted tree. |
| `results/stage3/search_summary.csv` | Evaluated tree configurations and their grouped validation scores. |
| `results/stage4/comparison.csv` | Five-model means for accuracy, macro precision, macro recall, macro-F1, gap, and fit time. |
| `results/stage4/comparison_folds.csv` | Fold metrics for the three trees and two forests. |
| `results/stage4/per_class_metrics.csv` | Pooled out-of-fold precision, recall, and F1 by class. |
| `results/stage4/oof_confusion_matrix.csv` | Tuned-forest pooled out-of-fold confusion counts. |
| `results/stage4/rf_search_summary.csv` | The 25 evaluated forests. |
| `results/stage5/forest_permutation_importance.csv` | Tuned-forest permutation and impurity importance for all 76 features. |
| `results/stage5/tree_permutation_importance.csv` | Pruned-tree permutation and impurity importance for all 76 features. |
| `results/stage5/correlated_pairs.csv` | Development-row feature pairs with absolute Spearman correlation of at least 0.7. |
| `results/stage5b/summary.csv` | Mean grouped validation accuracy, macro-F1, and per-class recall for configurations A, B, and C. |
| `results/stage5b/paired_differences.csv` | Fold-level differences from configuration A. |
| `results/stage5b/oof_confusion.csv` | Pooled out-of-fold confusion counts for both models and all three configurations. |
| `results/stage5b/fold_metrics.csv` | Fold accuracy, macro-F1, and per-class recall for both models and all three configurations. |
