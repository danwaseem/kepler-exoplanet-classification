# Person 3 — Danish Waseem

**Owns:** Decision Tree, Random Forest, pruning, feature importance, centroid sensitivity.

Existing stage modules remain at:

- `src/exoplanet/stage2.py` … `stage5b.py`
- `run_stage1.py` … `run_stage5b.py`
- `results/stage1/` … `results/stage5b/`
- `figures/stage2/` … `figures/stage5b/`

**Experiment wrappers:** `experiments/person3_trees/`

Trees and forests use `make_preprocess_pipeline(scale=False)` (median imputation only). Stages 1–5B are complete for the intermediate report. Do not retune selected hyperparameters or score the test partition without a separate decision.
