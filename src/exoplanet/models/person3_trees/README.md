# Person 3 — Danish Waseem

**Owns:** Decision Tree, Random Forest, pruning, feature importance, centroid sensitivity.

Existing stage modules remain at:

- `src/exoplanet/stage2.py` … `stage5b.py`
- `run_stage1.py` … `run_stage5b.py`
- Results: `results/person3/decision_tree/{stage2,stage3}/`, `random_forest/`, `feature_importance/`, `centroid_sensitivity/`
- Figures: matching paths under `figures/person3/`
- Shared Stage 1 outputs: `results/shared/`, `figures/shared/`, `docs/shared/`

**Experiment wrappers:** `experiments/person3_trees/`

Trees and forests use `make_preprocess_pipeline(scale=False)` (median imputation only). Stages 1–5B are complete for the intermediate report. Do not retune selected hyperparameters or score the test partition without a separate decision.
