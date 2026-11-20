# Person 1 experiments: baselines, Logistic Regression, Naive Bayes

Sanjana Reddy Katham. Reusable code is in `src/exoplanet/models/person1_baselines/`. Results go to `results/person1/`, figures to `figures/person1/`, and the written summary is `docs/person1/findings.md`.

All experiments use the shared development bundle (7,381 development rows with a measured transit depth, 76 proposed features), the saved host-star grouped `dev_fold` values, and median imputation (plus standardization where needed) fit inside each training fold. The test partition is never loaded for modeling or scored. Every `model_config.json` records `"test_scored": false` and the split SHA-256.

## Environment

The conda environment `data245` (Python 3.12) with the versions in the root `requirements.txt` (pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, joblib 1.6.0, pytest 9.2.0). scipy is used for the corrected paired t statistic and comes with scikit-learn.

## How to run

From the repository root, in order. Each step reads the saved outputs of the earlier ones and does not refit them.

```bash
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1

python experiments/person1_baselines/run_step1_e0_e3.py              # E0-E3 and nested CV
python experiments/person1_baselines/run_paired_comparisons.py       # paired fold-by-fold tests
python experiments/person1_baselines/run_step2_e4_calibration.py     # E4 calibration (binary)
python experiments/person1_baselines/run_step3_e5_temporal.py        # label crosstab and E5
python experiments/person1_baselines/run_step4_ranked_candidates.py  # ranked CANDIDATE list
python experiments/person1_baselines/make_figures.py                 # figures from saved results
pytest
```

The Step 1 to 3 scripts take `--n-jobs` (default `-1`, all cores). Parallelism is over folds and configurations, so results do not depend on it. Step 1 is the slow one, about 26 minutes on an 8-core laptop, because of L1 logistic regression with the saga solver at large C inside both the ordinary and the nested search. The other steps take seconds.

Model factories, the C grid and solver settings are in `person1_models.py` in this folder. They are not under `src/` because the shared test `tests/test_stage1.py::test_source_does_not_train_other_classifiers` rejects any file there that names the logistic regression class.

## What each step does

| Step | Script | Experiments | Main outputs |
| --- | --- | --- | --- |
| 1 | `run_step1_e0_e3.py` | E0 dummy floor; E1 logistic regression (no penalty, L2, L1, C from 0.001 to 1000); E2 Gaussian Naive Bayes (a) standardized, (b) signed log, (c) signed log plus Spearman pruning; E3 ablation (76, 56 without centroid, 76 plus vetting outputs as a diagnostic); nested CV for every selection | `results/person1/e0_*`, `e1_*`, `e2_*`, `e3_*`, `step1_summary.csv`, `step1_selection.json`, `e1_e2_grid.csv`, `e1_l1_sparsity.csv`, `e1_lr_coefficients.csv`, `e3_ablation.csv`, `nested_cv/`, `class_distribution/` |
| 1b | `run_paired_comparisons.py` | Paired fold-by-fold differences between Step 1 models from the saved fold metrics, with the corrected resampled t statistic | `results/person1/paired_comparisons.csv` |
| 2 | `run_step2_e4_calibration.py` | E4: Platt (sigmoid) and isotonic recalibration of the binary LR and NB (b), (c) with `CalibratedClassifierCV`, inner folds grouped by host star inside each training fold | `results/person1/e4_*`, `e4_calibration/` |
| 3 | `run_step3_e5_temporal.py` | Label crosstab between the old Kaggle copy and the current NASA table; E5 forward-in-time test on old CANDIDATEs that were later resolved | `results/person1/label_crosstab/`, `e5_temporal/`, `e5_reference_current_values_*` |
| 4 | `run_step4_ranked_candidates.py` | Calibrated binary model fit on all development CONFIRMED and FALSE POSITIVE rows, scoring the development CANDIDATEs | `results/person1/ranked_candidates_development.csv`, `ranked_candidates_summary.json` |
| 5 | `make_figures.py` | Report figures | `figures/person1/*.png` |

## Two setups

* `3class`: CONFIRMED, CANDIDATE, FALSE POSITIVE. Comparable with the other team members. Summaries for these runs come from the shared `cross_validate_estimator`; a second pass with the same factory collects probabilities, and the script checks that its out-of-fold labels are identical to the shared helper's.
* `binary`: CONFIRMED vs FALSE POSITIVE on the same folds, CANDIDATE rows dropped (5,876 rows). The shared helper averages macro scores over three fixed classes, so the binary runs use the fold loop in `cv.py` with the same `development_fold_masks`. Their `summary.json` has `"setup": "binary"`, two-class macro scores, and a `binary_confusion_matrix`. The 3x3 `confusion_matrix` and `oof_confusion_matrix.csv` keep the shared format and have an all-zero CANDIDATE row and column.

## Files in each experiment folder

| File | Contents |
| --- | --- |
| `fold_metrics.csv` | Per-fold train and validation accuracy, macro precision, recall, F1 |
| `summary.json` | Means, SDs, train vs validation gap, confusion matrix, split identity, setup |
| `per_class_metrics.csv` | Pooled out-of-fold precision, recall, F1, support |
| `oof_confusion_matrix.csv` | Pooled out-of-fold confusion counts |
| `model_config.json` | Hyperparameters, feature list, `test_scored: false`, split SHA |
| `extra_metrics.json` | Balanced accuracy, weighted P/R/F1, MCC, Cohen's kappa, per-class metrics, ROC-AUC, PR-AUC, Brier, log loss, ECE (binary): fold mean and SD, and pooled out of fold |
| `extra_fold_metrics.csv` | The same metrics per fold, plus convergence warnings and features used |
| `oof_predictions.csv.gz` | Cached out-of-fold labels and class probabilities, reused by later steps and figures |

`results/person1/reproduction_check.csv` compares these results with the run behind the intermediate report (largest difference 0.0002 on any shared metric; see `docs/person1/findings.md`).

`model_config.json` for the E3 leakage runs has `"diagnostic_only": true`. Those numbers show how much the vetting outputs encode the label. They are not model performance.

## Data file

`data/kaggle_cumulative.csv` is an older copy of the KOI cumulative table as distributed on Kaggle (9,564 rows, 50 columns, SHA-256 `b77f7332b7fe9fbc5ce4650b7d9f7aea035082ddd2b5701039eaadad2396db65`). It is used only for the label crosstab and E5. Only development-partition stars of the locked split are used for modeling.
