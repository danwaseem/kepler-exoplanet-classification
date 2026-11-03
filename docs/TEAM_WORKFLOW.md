# Team workflow

## Ownership

| Person | Models | Code | Experiments | Suggested branch |
| --- | --- | --- | --- | --- |
| 1 Sanjana Reddy Katham | Logistic Regression, Naive Bayes | `src/exoplanet/models/person1_baselines/` | `experiments/person1_baselines/` | `feature/logistic-naive-bayes` |
| 2 Venkata Sai Bhavana Yarrapragada | KNN, SVM | `src/exoplanet/models/person2_knn_svm/` | `experiments/person2_knn_svm/` | `feature/knn-svm` |
| 3 Danish Waseem | Decision Tree, Random Forest | `src/exoplanet/stage2.py`–`stage5b.py`, `models/person3_trees/` | `experiments/person3_trees/`, root `run_stage*.py` | `feature/decision-tree-random-forest` |
| 4 Nathan Pak | PCA, K-Means | `src/exoplanet/models/person4_pca_clustering/` | `experiments/person4_pca_clustering/` | `feature/pca-clustering` |

Shared modules: `config`, `io`, `audit`, `splits`, `preprocess`, `development`, `evaluation`.

## Common preprocessing rules

1. Use the proposed 76-feature list in `data/processed/proposed_leakage_aware_features.csv`.
2. Use the locked split in `data/processed/proposed_splits.csv` (SHA-256 `d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a`).
3. Train and validate only on `partition == development` primary rows (measured `koi_depth`). That is 7,381 rows.
4. Keep the 363 incomplete KOIs out of the primary experiment.
5. Fit median imputation inside each training fold. Use `make_preprocess_pipeline(scale=True)` for KNN, SVM, logistic regression, PCA, and K-Means. Use `scale=False` for trees and forests.
6. Do not fit preprocessing on the test partition.
7. Do not evaluate the test partition during development.
8. The feature list remains proposed until the team marks it final in `proposal_status.json`.

## Loading the shared development bundle

```python
from sklearn.pipeline import Pipeline
from exoplanet.development import load_development_bundle
from exoplanet.preprocess import make_preprocess_pipeline
from exoplanet.evaluation import cross_validate_estimator, write_experiment_results

bundle = load_development_bundle()

def factory():
    return Pipeline([
        *make_preprocess_pipeline(scale=True).steps,
        ("model", YourEstimator(...)),
    ])

summary = cross_validate_estimator(bundle, factory)
write_experiment_results("person2_knn_svm", "baseline_knn", summary, {"model": "KNN"})
```

## Expected model output format

Under `results/<owner>/<experiment_name>/`:

| File | Contents |
| --- | --- |
| `fold_metrics.csv` | Per-fold train/validation accuracy and macro-F1 |
| `summary.json` | Means, stds, gap, confusion matrix, split identity |
| `per_class_metrics.csv` | Pooled out-of-fold precision, recall, F1 |
| `oof_confusion_matrix.csv` | Pooled out-of-fold confusion counts |
| `model_config.json` | Hyperparameters, scaling flag, `test_scored: false` |

Report validation **macro-F1** as the primary comparison metric, with accuracy and per-class recall beside it.

## Integrating final comparisons

1. Each person merges development results into `main` through a pull request.
2. A later comparison notebook or script will read each owner's `summary.json`.
3. Final test evaluation happens once, after the team agrees, using the locked split and without refitting on test labels during model selection.
4. Person 3's Stages 1–5B metrics under `results/stage*/` are already the tree/forest development baseline.

## Centroid and diagnostic configurations

- Primary scientific set: 76 features (includes provisional centroid offsets).
- Sensitivity: 56 features without centroid offsets (`results/stage5b/`, configuration B).
- Diagnostic only: 76 features plus `koi_fwm_stat_sig` (configuration C). Not a production feature set.
