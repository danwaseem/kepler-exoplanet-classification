# Person 2 experiments — Venkata Sai Bhavana Yarrapragada

KNN and SVM on the shared development folds (grouped 5-fold CV, macro-F1). Features are standardized inside each training fold with `make_preprocess_pipeline(scale=True)`. The test partition is not scored.

| File | What it does |
| --- | --- |
| `KNN_SVM_Yarrapragada.ipynb` | Full exploration: data review, scaling experiment, KNN k sweep and grid search, SVM tuning, final comparison, larger-C and threshold checks, bias-variance analysis |
| `run_final_models.py` | Scores the default and tuned models with `cross_validate_estimator` and saves them with `write_experiment_results` |

```bash
python experiments/person2_knn_svm/run_final_models.py
```

| Output | Location |
| --- | --- |
| One folder per model (standard five files) | `results/person2/<experiment_name>/` |
| Combined table | `results/person2/summary.csv` |
| Notebook tables (grid searches, sweeps) | `results/person2/notebook/` |
| Figures | `figures/person2/` |

Model factories and final hyperparameters: `src/exoplanet/models/person2_knn_svm/models.py`.

Configurations follow `docs/shared/TEAM_WORKFLOW.md`: **A** = 76 features (primary), **B** = 56 features without centroid offsets (sensitivity). B is reported both retuned and with configuration-A hyperparameters (the approach used in `results/person3/centroid_sensitivity/`).
