# Person 1: Sanjana Reddy Katham

**Owns:** Logistic Regression, Naive Bayes, baseline evaluation, class distribution analysis.

**Experiment directory:** `experiments/person1_baselines/` (entry points, how to run, and the model factories in `person1_models.py`).

## Modules

| Module | Contents |
| --- | --- |
| `transforms.py` | `signed_log1p` and `SpearmanPruner`, the Naive Bayes feature repairs, fit inside each training fold |
| `data.py` | The two setups (three classes; binary CONFIRMED vs FALSE POSITIVE), feature sets (76, 56 without centroid columns, 76 plus the diagnostic leakage columns), and the join of the leakage columns by `kepoi_name` |
| `metrics.py` | Label and probability metrics the shared helper does not compute, fold summaries, and the corrected paired t statistic |
| `cv.py` | Fold loop on the saved development folds with out-of-fold probabilities, a parallel runner, summaries in the shared format, the result writer, and nested CV for model selection |
| `calibration.py` | `CalibratedClassifierCV` with inner folds grouped by host star and built from the training fold only, and reliability tables |
| `temporal.py` | Old vs current catalog join, label crosstab, and the E5 training and evaluation sets |

The model factories (`experiments/person1_baselines/person1_models.py`) are kept out of `src/` because `tests/test_stage1.py::test_source_does_not_train_other_classifiers` rejects any file under `src/` that names the logistic regression class.

## Shared API used

```python
from exoplanet.development import load_development_bundle, development_fold_masks
from exoplanet.preprocess import make_preprocess_pipeline, diagnostic_learning_frame
from exoplanet.evaluation import cross_validate_estimator, write_experiment_results
```

Preprocessing is fit inside each training fold. The test partition is not loaded or scored. Outputs are written under `results/person1/<experiment_name>/`.
