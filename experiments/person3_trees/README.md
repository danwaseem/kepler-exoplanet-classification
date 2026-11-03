# Person 3 experiments — Danish Waseem

Stages 1–5B already live at the repository root as `run_stage1.py` … `run_stage5b.py`. The wrappers in this directory call those same entry points so the ownership layout is clear for the team.

```bash
python experiments/person3_trees/run_stage4.py
```

| Stage | Results | Figures |
| --- | --- | --- |
| 1 (shared) | `results/shared/` | `figures/shared/` |
| 2–3 Decision Tree | `results/person3/decision_tree/stage2|stage3/` | `figures/person3/decision_tree/stage2|stage3/` |
| 4 Random Forest | `results/person3/random_forest/` | `figures/person3/random_forest/` |
| 5 Feature importance | `results/person3/feature_importance/` | `figures/person3/feature_importance/` |
| 5B Centroid sensitivity | `results/person3/centroid_sensitivity/` | `figures/person3/centroid_sensitivity/` |

Do not retune or score the test partition without a separate team decision.
