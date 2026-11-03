# Person 3 experiments — Danish Waseem

Stage 1 (shared) entry point: `scripts/run_stage1.py`. Stages 2–5B entry points live in this directory.

```bash
python scripts/run_stage1.py
python experiments/person3_trees/run_stage2.py
python experiments/person3_trees/run_stage3.py
python experiments/person3_trees/run_stage4.py
python experiments/person3_trees/run_stage5.py
python experiments/person3_trees/run_stage5b.py
```

| Stage | Results | Figures |
| --- | --- | --- |
| 1 (shared) | `results/shared/` | `figures/shared/` |
| 2–3 Decision Tree | `results/person3/decision_tree/stage2|stage3/` | `figures/person3/decision_tree/stage2|stage3/` |
| 4 Random Forest | `results/person3/random_forest/` | `figures/person3/random_forest/` |
| 5 Feature importance | `results/person3/feature_importance/` | `figures/person3/feature_importance/` |
| 5B Centroid sensitivity | `results/person3/centroid_sensitivity/` | `figures/person3/centroid_sensitivity/` |

Do not retune or score the test partition without a separate team decision.
