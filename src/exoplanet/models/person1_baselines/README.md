# Person 1 — Sanjana Reddy Katham

**Owns:** Logistic Regression, Naive Bayes, baseline evaluation, class distribution analysis.

**Experiment directory:** `experiments/person1_baselines/`

## Shared API

```python
from exoplanet.development import load_development_bundle
from exoplanet.preprocess import make_preprocess_pipeline
from exoplanet.evaluation import cross_validate_estimator, write_experiment_results
from sklearn.pipeline import Pipeline
```

Use `make_preprocess_pipeline(scale=True)` before logistic regression. Prefer the same within-fold scaler for Gaussian Naive Bayes. Fit preprocessing inside each training fold. Do not score the test partition during development.

Write outputs under `results/person1_baselines/<experiment_name>/`.
