# Person 2 — Venkata Sai Bhavana Yarrapragada

**Owns:** KNN, SVM, scaling, hyperparameter tuning.

**Experiment directory:** `experiments/person2_knn_svm/`

## Shared API

```python
from exoplanet.development import load_development_bundle
from exoplanet.preprocess import make_preprocess_pipeline
from exoplanet.evaluation import cross_validate_estimator, write_experiment_results
```

Always standardize after median imputation with `make_preprocess_pipeline(scale=True)`, fit inside the training fold only, and use the saved `dev_fold` values. Do not create a new split. Do not score the test partition during development.

Write outputs under `results/person2/<experiment_name>/`.
