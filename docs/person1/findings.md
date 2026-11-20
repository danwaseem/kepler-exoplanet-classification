# Person 1 findings: baselines, Logistic Regression, Naive Bayes, calibration, temporal validation

Sanjana Reddy Katham. This note summarizes what I ran, what I found, and what I learned. Every number below comes from a file under `results/person1/`, and the file is named next to each claim. Figures are in `figures/person1/`. How to rerun everything is in `experiments/person1_baselines/README.md`.

The dataset audit, the leakage-aware 76-feature proposal, the host-star grouped development/test split, the saved development folds, and the shared `src/exoplanet` loaders and evaluation helpers are Danish Waseem's work. I build on them and do not change them.

## Setup

* Data: the shared development bundle, 7,381 development KOIs with a measured transit depth, 6,357 host stars, 76 proposed features. Class counts are CONFIRMED 2,197, CANDIDATE 1,505, FALSE POSITIVE 3,679 (`results/person1/class_distribution/`, `figures/person1/class_distribution.png`).
* Folds: the five saved `dev_fold` values. No host star appears in both the training and validation part of any fold. The test partition was never loaded for modeling and never scored. Every `model_config.json` records `"test_scored": false` and the split SHA-256 `d0a2892a...b021a`.
* Two setups. The team setup has three classes. The vetting setup is binary, CONFIRMED vs FALSE POSITIVE, on the same folds with CANDIDATE rows removed (5,876 rows, 37.4% CONFIRMED).
* Preprocessing: median imputation and standardization from the shared `make_preprocess_pipeline`, fit inside each training fold. Naive Bayes variants add their own transforms between the imputer and the scaler, also inside the fold.
* Metrics: macro F1 is the primary comparison metric. Beside it I report accuracy, balanced accuracy, macro and weighted precision, recall and F1, per-class precision, recall and F1, Matthews correlation, Cohen's kappa, confusion matrices, the train minus validation macro F1 gap, ROC-AUC (one-vs-rest macro for three classes), PR-AUC, Brier score, log loss, and for the binary setup a 10-bin expected calibration error. Each is given as mean and SD over the five folds and once more pooled over all out-of-fold predictions (`extra_metrics.json` in each experiment folder). Log loss is marked not applicable for the dummy classifiers because their probabilities are exactly 0 or 1.
* Paired comparisons: all models share the same folds, so I compare them fold by fold. `results/person1/paired_comparisons.csv` gives the five per-fold differences, their mean and SD, how many folds favour each model, and the Nadeau-Bengio corrected resampled t statistic. With five folds the p-value is only a rough guide.

## Reproducing the earlier results

The code was ported from my intermediate-report scripts into the team structure. `results/person1/reproduction_check.csv` compares every metric the earlier run reported with the ported run. The largest absolute difference across macro F1, training macro F1, accuracy, ROC-AUC, PR-AUC, Brier and log loss is 0.0002. The one visible case is L1 logistic regression at C=10 in the three-class setup (0.7651 before, 0.7652 now), which comes from the higher iteration limit described under E1. The fold class counts and the 39 to 40 features kept by Spearman pruning also match.

Four E3 rows in that table were not rerun with their old settings. The wider C grid now selects a different logistic regression (L2 with C=1000 in both setups, where the earlier run used L2 with C=10 for three classes and the unpenalized model for binary), and E3 reuses whatever E1 selects. The new E3 values are within 0.002 of the old ones: three classes 0.719 vs 0.718 without centroid columns and 0.997 vs 0.998 with leakage columns; binary 0.917 vs 0.916 and 0.998 vs 0.998.

## E0: floor

| Setup | Model | Macro F1 | Accuracy |
| --- | --- | --- | --- |
| 3 classes | majority (always FALSE POSITIVE) | 0.222 | 0.498 |
| 3 classes | stratified random | 0.325 | 0.372 |
| binary | majority | 0.385 | 0.626 |
| binary | stratified random | 0.487 | 0.525 |

Source: `results/person1/step1_summary.csv`. An accuracy near 0.5 (three classes) or 0.63 (binary) means nothing by itself here. Every later model is read against these floors.

## E1: logistic regression

Unpenalized, L2 and L1, all with `class_weight="balanced"`. C was searched over 0.001, 0.01, 0.1, 1, 10, 100 and 1000 (`results/person1/e1_e2_grid.csv`).

| Setup | Model | Macro F1 (SD) | Train minus validation gap | ROC-AUC | PR-AUC | Brier | Log loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 classes | no penalty | 0.771 (0.013) | 0.016 | 0.924 | 0.811 macro | 0.291 | 0.542 |
| 3 classes | L2, C=1000 | 0.772 (0.014) | 0.015 | 0.924 | 0.810 macro | 0.291 | 0.542 |
| 3 classes | L1, C=1000 | 0.766 (0.011) | 0.013 | 0.922 | 0.808 macro | 0.296 | 0.528 |
| binary | no penalty | 0.951 (0.004) | 0.008 | 0.988 | 0.973 | 0.035 | 0.151 |
| binary | L2, C=1000 | 0.952 (0.004) | 0.007 | 0.988 | 0.973 | 0.035 | 0.151 |
| binary | L1, C=10 | 0.948 (0.003) | 0.006 | 0.988 | 0.974 | 0.038 | 0.152 |

Three-class Brier is the multiclass version (sum over classes, range 0 to 2). Source: `step1_summary.csv`, `e1_*` folders.

* The selected L2 model, C=1000, sits at the top of the grid in both setups. The earlier grid stopped at C=10, which was also the chosen value there, so I extended it by two decades. The search again picks the largest value. That does not mean a larger C would find a better model. Larger C means a weaker penalty, and the limit as C grows is the unpenalized model, which is already in the comparison. L2 macro F1 for C = 10, 100 and 1000 is 0.7713, 0.7710 and 0.7721 (three classes) and 0.9499, 0.9515 and 0.9516 (binary). The unpenalized model gets 0.7712 and 0.9511. Paired over the five folds, L2 C=1000 minus no penalty is +0.0008 (SD 0.0007, corrected p = 0.15) for three classes and +0.0005 (SD 0.0010, better in 3 of 5 folds, p = 0.47) for binary. Those gaps are a tenth of the fold SD. The two models also have the same ROC-AUC (0.924 and 0.988) and almost the same train minus validation gap. Their standardized coefficients correlate at 0.9988 (three classes) and 0.99997 (binary), and in the binary model the largest difference between any pair of coefficients is 0.25, on weights up to 20 in size (`e1_lr_coefficients.csv`).
* So regularization does not help here. The curve rises up to about C=10 and is flat after that, and stronger penalties (smaller C) only cost accuracy: C=0.001 gives 0.734 and 0.895. With about 4,700 to 5,900 training rows per fold and 76 standardized features, the unpenalized fit is already stable. "C=1000 selected" and "no penalty" should be read as the same answer. I keep the L2 C=1000 label because that is what the selection rule returned. The nested CV below confirms it: the inner folds choose C=100, C=1000 or the unpenalized model in every outer fold.
* Overfitting is small. The train minus validation macro F1 gap is 0.006 to 0.016. With 76 features and about 4,700 to 5,900 training rows per fold, a linear model has little room to overfit.
* L1 sparsity (`e1_l1_sparsity.csv`, `figures/person1/l1_sparsity.png`). At C=0.001 L1 keeps about 2 features and loses most of its accuracy (three classes 0.575, binary 0.746). At C=0.01 it keeps 25.4 (three classes) or 17.2 (binary) features at a cost of about 0.03 macro F1. At the selected C it keeps all 76. A compact model is possible, but nothing in the score rewards it.
* Convergence. In the earlier run the saga solver for L1 stopped at its 5,000-iteration limit on several folds. A test fold at C=1000 needed about 4,800 iterations at the default tolerance, so I raised the limit to 20,000 and kept the tolerance at 1e-4. No convergence warnings remain in any fold or any C (`convergence_warnings` in `extra_fold_metrics.csv`).
* Coefficients (`e1_lr_coefficients.csv`, `figures/person1/lr_coefficients_binary.png`). In the binary L2 model on standardized features, the largest weights push toward FALSE POSITIVE: transit depth (-20.0), impact parameter (-12.9), the lower impact uncertainty (-7.6) and the planet-to-star radius ratio (-6.6). Deep and grazing transits are what an eclipsing binary looks like, so the signs make physical sense. The upper radius ratio uncertainty has a large positive weight (+6.9) that offsets the radius ratio itself. That is a sign of correlated features sharing weight, so individual coefficients should not be read one at a time.
* Per class (pooled out of fold, L2 C=1000, three classes): recall is 0.883 for CONFIRMED, 0.721 for CANDIDATE and 0.767 for FALSE POSITIVE. Most errors involve CANDIDATE. 729 false positives are predicted as CANDIDATE (`e1_lr_l2_3class/oof_confusion_matrix.csv`, `figures/person1/confusion_matrices.png`). That is expected: CANDIDATE means "not yet decided", not a physical class.

### Nested cross-validation for choosing C

Choosing C on the same five folds that are reported makes the reported score slightly optimistic. To measure how much, I reran every selection inside each outer training fold: four inner folds grouped by host star, built only from the outer training rows, choose the configuration; the chosen configuration is then scored on the untouched outer fold. Its outer score is the same fit as in the ordinary run, so it is reused rather than refit (`results/person1/nested_cv/`).

| Setup | What is being chosen | Non-nested | Nested | Optimism |
| --- | --- | --- | --- | --- |
| 3 classes | L2: C | 0.7721 | 0.7719 | 0.0001 |
| 3 classes | L1: C | 0.7658 | 0.7658 | 0.0000 |
| 3 classes | LR: penalty and C | 0.7721 | 0.7717 | 0.0004 |
| 3 classes | NB variant | 0.6699 | 0.6699 | 0.0000 |
| 3 classes | LR or NB, everything | 0.7721 | 0.7717 | 0.0004 |
| binary | L2: C | 0.9516 | 0.9513 | 0.0003 |
| binary | L1: C | 0.9478 | 0.9478 | 0.0000 |
| binary | LR: penalty and C | 0.9516 | 0.9513 | 0.0004 |
| binary | NB variant | 0.9159 | 0.9159 | 0.0000 |
| binary | LR or NB, everything | 0.9516 | 0.9513 | 0.0004 |

The optimism is at most 0.0004 macro F1, far below the fold SD (0.004 to 0.014). The inner folds pick C=100 or C=1000 for L2, or the unpenalized model, in every outer fold, and they pick NB (b) in every outer fold. The selection is stable because the score surface is flat at large C. In the earlier report I described the non-nested scores as "slightly optimistic" without a number. The number is now measured and negligible.

## E2: Gaussian Naive Bayes

| Variant | 3-class macro F1 | Binary macro F1 | Binary PR-AUC | Binary log loss | Features used |
| --- | --- | --- | --- | --- | --- |
| (a) standardized | 0.574 | 0.878 | 0.801 | 3.69 | 76 |
| (b) signed log `sign(x)*log1p(abs(x))`, then standardized | **0.670** | **0.916** | 0.935 | 1.77 | 76 |
| (c) signed log, drop one of each pair with abs(Spearman r) > 0.8 inside the training fold | 0.642 | 0.892 | **0.963** | **1.49** | 39 (3 classes), 40.2 (binary) |

Source: `step1_summary.csv`, `e2_*` folders, `paired_comparisons.csv`.

* The log transform is the largest single repair. Gaussian NB assumes each feature is normal within a class. Depth, SNR, period, insolation and most uncertainties are heavy-tailed, so on the raw scale a few extreme values dominate the class likelihoods. The transform adds +0.096 macro F1 (three classes, all five folds better, corrected p = 0.0008) and +0.038 (binary, all five folds, p = 0.005).
* Pruning correlated features costs macro F1 at the default 0.5 threshold (-0.028 three classes, -0.024 binary, all five folds) but improves ranking and probability quality (binary PR-AUC 0.963 vs 0.935, log loss 1.49 vs 1.77). Which variant is "better" depends on the metric. That is why calibration was the next step.
* NB trails logistic regression by 0.102 macro F1 in three classes and by 0.036 in binary, with LR ahead in all five folds of both setups. Without calibration, NB probabilities are very overconfident: binary log loss 1.77 against 0.15 for LR.
* NB recall in three classes is high for CONFIRMED (0.939) and low for CANDIDATE (0.527). 1,162 false positives are called CANDIDATE.

## E3: centroid ablation and leakage diagnostic

The selected LR and NB per setup, retrained on three feature sets (`results/person1/e3_ablation.csv`, `figures/person1/centroid_ablation.png`).

| Setup | Model | 56 features (no centroid) | 76 features (primary) | 76 + vetting outputs (diagnostic only) |
| --- | --- | --- | --- | --- |
| 3 classes | LR, L2 C=1000 | 0.719 | 0.772 | 0.997 |
| 3 classes | NB (b) | 0.604 | 0.670 | 0.999 |
| binary | LR, L2 C=1000 | 0.917 | 0.952 | 0.998 |
| binary | NB (b) | 0.830 | 0.916 | 0.999 |

* The 20 provisional centroid columns are worth 0.035 to 0.086 macro F1, and the drop is seen in all five folds for both models and both setups (`paired_comparisons.csv`). They carry real signal. Whether that signal is acceptable or is leakage through the vetting process is a team decision about the feature list. These numbers show how much depends on it.
* The diagnostic adds `koi_score`, `koi_pdisposition` (as a CANDIDATE indicator), the four false-positive flags, and whether a Kepler name exists. Both models become almost perfect (0.997 to 0.999). This is not performance. It confirms that these columns encode the label and must stay out of any real model. Each of these runs has `"diagnostic_only": true` in its `model_config.json`.
* The E3 runs reuse the configuration chosen on 76 features and are not re-tuned on 56. Given how flat the C curve is, re-tuning would change little.

## E4: probability calibration (binary)

The selected binary LR and NB variants (b) and (c) were recalibrated with Platt scaling (sigmoid) and isotonic regression using `CalibratedClassifierCV`. Its inner folds are four host-star grouped folds built from the outer training fold only, so the outer validation fold is never used to fit either the classifier or the calibrator (`results/person1/e4_calibration/`, `e4_*` folders, `figures/person1/reliability_diagrams.png`).

| Model | Variant | Brier | Log loss | ECE | ROC-AUC | PR-AUC | Macro F1 | Mean P(CONFIRMED) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR, L2 C=1000 | uncalibrated | 0.0353 | 0.151 | 0.021 | 0.988 | 0.973 | 0.952 | 0.390 |
| | sigmoid | 0.0365 | 0.145 | 0.037 | 0.989 | 0.978 | 0.953 | 0.374 |
| | isotonic | 0.0340 | 0.129 | 0.015 | 0.989 | 0.977 | 0.953 | 0.374 |
| NB (b) | uncalibrated | 0.0791 | 1.765 | 0.080 | 0.971 | 0.935 | 0.916 | 0.429 |
| | sigmoid | 0.0667 | 0.239 | 0.021 | 0.972 | 0.942 | 0.918 | 0.374 |
| | isotonic | 0.0433 | 0.167 | 0.016 | 0.979 | 0.970 | 0.944 | 0.374 |
| NB (c) | uncalibrated | 0.1003 | 1.489 | 0.102 | 0.978 | 0.963 | 0.892 | 0.457 |
| | sigmoid | 0.0791 | 0.269 | 0.032 | 0.978 | 0.965 | 0.900 | 0.375 |
| | isotonic | 0.0469 | 0.169 | 0.014 | 0.978 | 0.961 | 0.935 | 0.375 |

Fold means from `e4_calibration/before_after.csv`. The observed CONFIRMED share is 0.374.

* Naive Bayes needs calibration and gets a lot from it. Isotonic cuts NB (b) log loss from 1.77 to 0.17 and Brier from 0.079 to 0.043, better in all five folds (corrected p = 0.002 and 0.004). It also raises NB (b) macro F1 at the 0.5 threshold from 0.916 to 0.944, because the uncalibrated probabilities put the threshold in the wrong place. Calibrated NB (b) comes within 0.008 of LR.
* Logistic regression is already close to calibrated. Isotonic lowers its Brier from 0.0353 to 0.0340 and its log loss from 0.151 to 0.129, improving in all five folds, but the change is small (corrected p = 0.06 and 0.24). Platt scaling makes the LR's ECE worse (0.021 to 0.037), because the reliability curve of the balanced LR is not sigmoid shaped.
* Uncalibrated, the balanced LR over-predicts CONFIRMED (mean probability 0.390 vs 0.374 observed). That is the expected effect of `class_weight="balanced"`, which reweights the classes as if they were equally common. Calibration brings the mean back to the observed share.
* The reliability diagrams show the shape of the problem. Uncalibrated NB puts almost everything near 0 or 1, and in the middle bins the observed share is far from the prediction. Isotonic follows the diagonal for all three models.
* The isotonic-calibrated LR has the lowest pooled Brier score (0.0340) and was selected for the ranked CANDIDATE list (`e4_calibration/selected_calibrated_model.json`).

## Label drift between catalog versions

I compared `koi_disposition` in an older Kaggle copy of the KOI cumulative table (`experiments/person1_baselines/data/kaggle_cumulative.csv`) with the NASA 2026-10-06 table. The two files have the same 9,564 KOIs with matching `kepid` (`results/person1/label_crosstab/`, `figures/person1/label_crosstab.png`).

* 925 labels changed. Of the old CANDIDATE objects, 591 were resolved: 443 to CONFIRMED and 148 to FALSE POSITIVE. 320 old FALSE POSITIVE objects are CANDIDATE now, 13 old FALSE POSITIVE are now CONFIRMED, and one old CONFIRMED is now FALSE POSITIVE. These match the earlier comparison exactly.
* In the development partition alone, 725 labels changed and 468 old candidates were resolved (353 to CONFIRMED, 115 to FALSE POSITIVE).
* Labels in this catalog are not fixed ground truth. They move in both directions, including from FALSE POSITIVE back to CANDIDATE.

## E5: temporal validation

The question: if I had trained a model on the old catalog, how well would it have predicted the way its open candidates were later resolved? (`results/person1/e5_temporal/`, `figures/person1/e5_temporal_validation.png`)

Design:

* Training rows are old-catalog CONFIRMED and FALSE POSITIVE labels with old feature values. Evaluation rows are old CANDIDATE objects that are now CONFIRMED or FALSE POSITIVE, also with their old feature values. Nothing from the current table enters the model except the evaluation labels.
* Only stars in the development partition of the locked split are used. The 1,911 rows on test-partition stars are dropped before anything else.
* Rows need a measured transit depth in the old file (272 development rows dropped), matching the primary experiment's rule.
* Training rows on any star that hosts an evaluation object are removed (110 rows), so no host star is on both sides.
* Features are the 29 of the 76 that exist in the old file. The 47 lost include all 20 centroid columns, the KIC magnitudes, `koi_count`, the radius ratio, scaled semi-major axis, inclination, stellar metallicity, mass and density, and their uncertainties (`e5_temporal/features.json`).
* Result: 5,515 training rows (1,750 CONFIRMED, 3,765 FALSE POSITIVE) and 464 evaluation rows (352 now CONFIRMED, 112 now FALSE POSITIVE) (`e5_temporal/counts.json`).
* Why 464 and not 468. The development partition has 468 old candidates that were later resolved (353 to CONFIRMED, 115 to FALSE POSITIVE; `label_crosstab/summary.json`). Four of them have no transit depth in the old file: K02410.01, K02410.02 and K00005.02, now FALSE POSITIVE, and K07368.01, now CONFIRMED. They also lack planet radius, SNR and stellar parameters there, and they have no depth in the current table either, so they belong to the 363 incomplete KOIs that the primary experiment leaves out. Applying the same measured-depth rule removes them: 353 - 1 = 352 and 115 - 3 = 112. No evaluation row is lost for any other reason.

| Model | Macro F1 | Accuracy | Recall CONFIRMED | Recall FALSE POSITIVE | ROC-AUC | PR-AUC | Brier | Precision in top 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR, L2 C=1000 | **0.862** | **0.897** | 0.920 | 0.821 | 0.917 | 0.948 | **0.092** | 0.98 |
| NB (b) | 0.834 | 0.881 | 0.935 | 0.714 | 0.917 | **0.970** | 0.116 | **1.00** |
| LR + isotonic calibration | 0.790 | 0.823 | 0.804 | 0.884 | **0.922** | 0.955 | 0.127 | 0.98 |
| always planet | 0.431 | 0.759 | 1.000 | 0.000 | 0.500 | 0.759 | 0.241 | 0.82 |

Source: `e5_temporal/temporal_metrics.csv`.

* All models beat the always-planet predictor by a wide margin on everything except its own strength, CONFIRMED recall. Its 0.759 accuracy shows why accuracy alone is misleading here: about three quarters of the resolved candidates became planets.
* Ranking holds up well. 98 of the LR's 100 highest-scored old candidates are now CONFIRMED (NB: 100 of 100), against a base rate of 0.76.
* Two references separate the causes of the drop (`e5_temporal/reference_metrics.csv`). With the same 29 features, the LR reaches 0.895 macro F1 and 0.965 ROC-AUC on the current table's development folds, and 0.888 and 0.963 in grouped CV inside the old training rows. The 76 to 29 feature loss explains most of the gap to the main E1 result (0.952). The step forward in time costs a further 0.03 macro F1 and 0.05 ROC-AUC. Candidates are harder than objects that were already resolved when the old catalog was made.
* Calibration does not transfer. The calibrated LR has the best ROC-AUC but the worst Brier score and macro F1 of the three models. The calibrator learned the training prior (31.7% CONFIRMED), while 75.9% of the resolved candidates became planets. Its mean predicted P(CONFIRMED) on the evaluation set is 0.596, against 0.759 observed. The uncalibrated balanced LR (0.665) and NB (0.774) are closer by accident of their own biases. Calibration fixes probabilities for the population it was fit on, not for a population with a different class mix.
* Sensitivity check: dropping the 214 training rows whose old label has since changed barely moves anything. LR macro F1 is 0.854 instead of 0.862 and ROC-AUC 0.915 instead of 0.917. The calibrated LR improves slightly (Brier 0.123 vs 0.127). The label changes in the training set are too few to matter.

## Ranked CANDIDATE list

The isotonic-calibrated LR was fit on all 5,876 development CONFIRMED and FALSE POSITIVE rows (76 features, calibrator inner folds grouped by host star) and used to score the 1,505 development CANDIDATE rows (`results/person1/ranked_candidates_development.csv`, `ranked_candidates_summary.json`, `figures/person1/ranked_candidates_histogram.png`).

* 238 candidates score at least 0.9, 303 score 0.5 to 0.9, 361 score 0.1 to 0.5, and 603 score below 0.1. The sum of the calibrated probabilities is 564, an estimate of how many would be confirmed if the calibration held for candidates.
* E5 shows it probably does not hold. Candidates resolved forward in time were confirmed more often than the training prior suggests. The probabilities are best read as a ranking for follow-up, and the expected count as a lower estimate.
* Isotonic calibration is a step function, so many candidates share a value (614 distinct values among 1,505). Ties are broken by the uncalibrated probability of the same model. The two orderings have Spearman correlation 0.997.
* 11 candidates have at least one feature outside the range seen in the training rows. The list records how many and which ones. The top-ranked object, K01174.01, has a period of about 130,000 days, far beyond the Kepler baseline, and is out of range on 7 orbit-derived features. Its near-1.0 score is an extrapolation by a linear model and should be checked by hand. It is the only out-of-range object in the high band.

## Difficulties

* The shared helper fixes the three-class label order. Its macro scores average over CONFIRMED, CANDIDATE and FALSE POSITIVE, so a binary run passed through it would count a CANDIDATE F1 of 0 and report about 0.63 instead of 0.95. I wrote a separate fold loop for the binary setup that uses the same saved folds and writes the same files through `write_experiment_results`. Its summaries mark the setup and give the two-class confusion matrix. For the three-class runs I kept the shared helper as the source of the reported summaries and checked that my probability loop gives identical out-of-fold labels.
* The shared feature guard refuses the vetting columns, by design. The leakage diagnostic therefore runs through my own fold loop, with the columns attached from the raw table by `kepoi_name`. The bundle's row index is the position in the processed matrix, not in the raw table, so an index join would have silently misaligned rows. Matching on `kepoi_name`, with an equality check, avoided that.
* A shared test (`tests/test_stage1.py::test_source_does_not_train_other_classifiers`) rejects any file under `src/` that names the logistic regression class. My model factories therefore live in `experiments/person1_baselines/person1_models.py`. The parts that do not name an estimator stay in `src/exoplanet/models/person1_baselines/`.
* L1 logistic regression with the saga solver is slow at large C in the three-class setup (about 30 seconds per fit), and nested CV multiplies the number of fits. Running every fold and configuration as a separate parallel task brought all of Step 1 to about 26 minutes on 8 cores.
* With per-fold calibrators, pooled out-of-fold ranking metrics can be misleading. Each fold's Platt mapping is different, so pooling their outputs mixes five scales. NB (b) with sigmoid calibration has a fold-mean ROC-AUC of 0.972 but a pooled ROC-AUC of 0.946. I report fold means as the main numbers and keep pooled values beside them.
* The old Kaggle file has only 29 of the 76 features, so E5 measures a reduced model. The two reference runs were added so the feature loss and the step forward in time could be told apart.

## Lessons learned

* Check the floor first. A 0.63 binary accuracy is what always guessing FALSE POSITIVE gets. Macro F1, balanced accuracy and MCC make that obvious where accuracy does not.
* A tuned hyperparameter at the edge of the grid is worth chasing, but the answer can be "no regularization needed". The extended grid confirmed that the data are large enough relative to 76 features that the penalty does nothing useful.
* Measure optimism instead of describing it. Nested CV took about 13 extra minutes of compute and replaced "slightly optimistic" with "at most 0.0004".
* Fixing a model's assumptions matters more than switching models. The signed log transform gave Naive Bayes +0.10 macro F1, and calibration gave it another +0.03 and a log loss about ten times lower.
* Calibration is about a population, not a model. It fixed NB on the development folds and made LR worse on the forward-in-time candidates, because the class mix changed.
* Catalog labels move. 925 of 9,564 changed between two versions, in both directions. A model trained on one version is being graded partly on the archive's later work.
* Strong features can be strong for the wrong reason. Adding the vetting outputs gave 0.998 macro F1, which is a warning sign rather than a result. The centroid columns sit in between, and their value depends on a team decision about what counts as leakage.

## Best practices I followed

* Keep every fitted step (imputer, log transform, pruning, scaler, calibrator) inside a Pipeline fit on the training fold only, including inner folds for nested CV and calibration.
* Group every split by host star, at every level: outer folds, inner folds, calibration folds, and the E5 train/evaluation separation.
* Never load the test partition during development. Record `test_scored: false` and the split checksum with every result.
* Compare models on the same folds and report paired differences, not two separate means.
* Report label metrics and probability metrics together, as fold means with SDs and pooled out of fold.
* Treat columns that encode the label as a diagnostic and label those results clearly.
* Cache out-of-fold predictions and build later steps and figures from saved files, so nothing is refit without need.
* Port old code against a written reproduction check and explain every difference.

## How these results relate to published Kepler vetting work

Published Kepler vetting systems are not directly comparable with these numbers, for three reasons. They usually work from light curves or from the full set of Robovetter metrics. They often predict the Robovetter's own pass/fail call rather than the archive disposition. And their reported scores come from different catalog versions and splits. Here, the Robovetter outputs (`koi_score`, the false-positive flags, `koi_pdisposition`) are deliberately excluded, and E3 shows what including them does. The fair claims are narrower. With catalog-level transit, stellar and centroid parameters only, a linear model separates CONFIRMED from FALSE POSITIVE at macro F1 0.952 and ROC-AUC 0.988 on host-star grouped folds. Trained on an older catalog with 29 features, it ranks later-resolved candidates at ROC-AUC 0.917.

Work to place this against (citations to be checked against the papers before they go in the report): the DR24 and DR25 Robovetter papers (Coughlin et al. 2016; Thompson et al. 2018), the random-forest Autovetter (McCauliff et al. 2015), the convolutional-network vetting of Shallue and Vanderburg (2018), and machine-learning validation of Kepler candidates (Armstrong et al. 2021). For methods: Platt (1999) for sigmoid calibration, Zadrozny and Elkan (2002) for isotonic calibration, Niculescu-Mizil and Caruana (2005) on Naive Bayes probabilities being pushed toward 0 and 1, and Nadeau and Bengio (2003) for the corrected resampled t statistic.

The Kaggle file is a dataset upload, not a competition, so there is no leaderboard to place this work on.

## Figures

| File | Shows |
| --- | --- |
| `class_distribution.png` | Development class counts and shares |
| `confusion_matrices.png` | Pooled out-of-fold confusion matrices, best LR and NB, both setups |
| `lr_vs_nb_comparison.png` | Macro F1 of every E0 to E2 model with per-fold points, both setups |
| `centroid_ablation.png` | E3: 56 vs 76 features vs the leakage diagnostic |
| `lr_coefficients_binary.png` | Largest standardized coefficients of the binary L2 model, mean and SD over folds |
| `l1_sparsity.png` | Features kept and macro F1 against C for L1 |
| `reliability_diagrams.png` | Reliability before and after Platt and isotonic calibration |
| `e5_temporal_validation.png` | E5 forward-in-time results against the two references and the always-planet line |
| `label_crosstab.png` | Old vs current labels for all 9,564 KOIs |
| `ranked_candidates_histogram.png` | Distribution of calibrated scores for the development CANDIDATEs |
