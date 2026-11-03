# Person 4 — Nathan Pak

**Owns:** PCA, K-Means, dimensionality reduction, clustering visualization.

**Experiment directory:** `experiments/person4_pca_clustering/`

## Shared API

```python
from exoplanet.development import load_development_bundle
from exoplanet.preprocess import make_preprocess_pipeline
```

Standardize with `make_preprocess_pipeline(scale=True)` inside the training fold before PCA or K-Means. Do not fit either method on the test partition. Use the saved development folds for any held-out assessment of reconstruction or cluster stability.

Write outputs under `results/person4_pca_clustering/<experiment_name>/` and figures under `figures/person4_pca_clustering/`.
