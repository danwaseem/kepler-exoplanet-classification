# Exoplanet Classification Using Machine Learning

DATA 245 group project at San José State University.

**Objective:** Compare classical machine learning algorithms for classifying NASA Kepler Objects of Interest (KOI) as CONFIRMED, CANDIDATE, or FALSE POSITIVE.

This repository is a shared team workspace. Four members implement different methods in parallel, using the same dataset, feature proposal, host-star grouped splits, and evaluation helpers. Individual model results are not the final project outcome until the group compares them and agrees on next steps.

GitHub: [danwaseem/kepler-exoplanet-classification](https://github.com/danwaseem/kepler-exoplanet-classification)

## What the group will investigate

- Supervised classification performance across several classical algorithms
- Overfitting and regularization (for example tree pruning and model complexity)
- Ensemble learning (Random Forest compared with a single Decision Tree)
- Feature importance and scientific interpretability
- Data leakage controls in the KOI cumulative table
- Dimensionality reduction (PCA)
- Unsupervised clustering (K-Means)
- Fair comparison on locked development folds before any final test evaluation

## Team responsibilities

| Person | Name | Technical ownership | Experiment package |
| --- | --- | --- | --- |
| 1 | Sanjana Reddy Katham | Logistic Regression, Naive Bayes, baselines, class distribution | `experiments/person1_baselines/`, `src/exoplanet/models/person1_baselines/` |
| 2 | Venkata Sai Bhavana Yarrapragada | KNN, SVM, scaling, hyperparameter tuning | `experiments/person2_knn_svm/`, `src/exoplanet/models/person2_knn_svm/` |
| 3 | Danish Waseem | Decision Tree, Random Forest, pruning, feature importance, centroid sensitivity | `experiments/person3_trees/`, `src/exoplanet/stage2.py`–`stage5b.py`, `scripts/run_stage1.py` |
| 4 | Nathan Pak | PCA, K-Means, dimensionality reduction, clustering visualization | `experiments/person4_pca_clustering/`, `src/exoplanet/models/person4_pca_clustering/` |

All four members share responsibility for dataset understanding, cleaning and preprocessing, exploratory analysis, methodology, evaluation design, final model comparison, interpretation, documentation, and the final presentation.

Each person writes results under `results/personN/`, figures under `figures/personN/`, and notes under `docs/personN/`. Do not edit another person’s directories. Shared preprocessing and evaluation live in `src/exoplanet/` outside the per-person `models/` packages. See `docs/shared/TEAM_WORKFLOW.md` and `CONTRIBUTING.md`.

## Current project status

### Completed

- Shared NASA KOI snapshot, column audit, and leakage-aware feature proposal (76 columns; still **proposed** until the team finalizes it)
- Locked host-star grouped development / test split and development CV folds
- Shared loaders, median-imputation pipelines, development bundle API, and evaluation helpers
- Stage 1 documentation and EDA figures under `results/shared/`, `figures/shared/`, and `docs/shared/`
- Person 3 workstream (Decision Tree and Random Forest): overfitting baseline, pruning, tuned forest comparison, feature importance, and centroid sensitivity on development folds only

Selected Person 3 development validation macro-F1 (not a final team result): pruned tree about 0.7716; tuned forest about 0.8156. Details: `docs/shared/experimental_summary.md` and `docs/person3/`.

### In progress

- Person 1: Logistic Regression and Naive Bayes experiments
- Person 2: KNN and SVM experiments
- Person 4: PCA and K-Means experiments
- Team review of the proposed 76-feature list

### Planned

- Combine development results across all four workstreams
- Agree whether to finalize features or run additional sensitivity analyses
- Single agreed evaluation of the held-out test partition (not opened during individual development)
- Incomplete-KOI sensitivity experiment, if the team decides to include it
- Final report, presentation, and demonstration

Do not claim other teammates’ models are complete. Do not score the test partition until the team agrees.

## Dataset

- Source: NASA Exoplanet Archive, Kepler Objects of Interest Cumulative Table
- DOI: [10.26133/NEA4](https://doi.org/10.26133/NEA4)
- Filename version date: 2026-10-06
- Canonical file: `data/raw/koi_cumulative_2026-10-06.csv`
- SHA-256: `a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab` (11,843,781 bytes)
- Shape: 9,564 rows × 153 columns

Do not replace this CSV with a newer NASA download. If the file is missing, restore the exact snapshot and verify the checksum above.

Locked split: `data/processed/proposed_splits.csv`  
SHA-256: `d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a`

Primary features: 76 leakage-aware columns in `data/processed/proposed_leakage_aware_features.csv`. Primary modeling uses 7,381 development rows with a measured transit depth. The 363 incomplete KOIs remain in the split file and are excluded from the primary experiment.

Attribution: This research has made use of the NASA Exoplanet Archive, which is operated by the California Institute of Technology, under contract with the National Aeronautics and Space Administration under the Exoplanet Exploration Program.

## Setup

```bash
git clone https://github.com/danwaseem/kepler-exoplanet-classification.git
cd kepler-exoplanet-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the dataset, split, feature list, and shared API:

```bash
python team_shared_setup/verify_shared_setup.py
pytest
```

Optional checksum check:

```bash
PYTHONPATH=src python - <<'PY'
from exoplanet.io import file_sha256
from exoplanet.config import RAW_CSV, DATA_PROCESSED, SAVED_SPLIT_SHA256
assert file_sha256(RAW_CSV) == "a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab"
assert file_sha256(DATA_PROCESSED / "proposed_splits.csv") == SAVED_SPLIT_SHA256
print("checksums ok")
PY
```

## Shared preprocessing and evaluation

Run scripts from the repository root so paths resolve correctly. Use the saved `dev_fold` assignments. Fit imputation (and scaling when needed) only on each training fold. Do not score the test partition during development. Report grouped validation **macro-F1** as the primary comparison metric, with accuracy and per-class recall beside it.

```python
from exoplanet.development import load_development_bundle
from exoplanet.preprocess import make_preprocess_pipeline
from exoplanet.evaluation import cross_validate_estimator, write_experiment_results
from sklearn.pipeline import Pipeline

bundle = load_development_bundle()  # 7381 development rows, 76 features

def factory():
    return Pipeline([
        *make_preprocess_pipeline(scale=True).steps,  # use scale=False for trees / forests
        ("model", YourModel(...)),
    ])

summary = cross_validate_estimator(bundle, factory)
write_experiment_results("person2", "baseline", summary, {"model": "..."})
```

Teammates do not need Person 3 stage scripts to train their own models. The shared API above is enough for independent work.

## Running experiments

Add code under your owned directories, then:

```bash
PYTHONPATH=src python experiments/person1_baselines/your_script.py
PYTHONPATH=src python experiments/person2_knn_svm/your_script.py
PYTHONPATH=src python experiments/person4_pca_clustering/your_script.py
```

Person 3 stages are already completed. Re-running them should not overwrite results without a team decision:

```bash
python scripts/run_stage1.py
python experiments/person3_trees/run_stage2.py
python experiments/person3_trees/run_stage3.py
python experiments/person3_trees/run_stage4.py
python experiments/person3_trees/run_stage5.py
python experiments/person3_trees/run_stage5b.py
```

Save new metrics under `results/personN/<experiment_name>/`, figures under `figures/personN/`, and notes under `docs/personN/`.

## Repository layout

```
data/raw/                 NASA CSV (canonical)
data/processed/           features, splits, matrices
data/external/            column-page documentation
src/exoplanet/            shared loaders, preprocess, evaluation
src/exoplanet/models/     per-person model packages
experiments/              person1_baselines, person2_knn_svm, person3_trees, person4_pca_clustering
scripts/                  shared Stage 1 entry point
results/shared/           Stage 1 audits and shared artifacts
results/person1…4/        per-person metrics
figures/shared/           shared EDA figures
figures/person1…4/        per-person plots
docs/shared/              dataset, Stage 1, workflow, experimental summary
docs/person1…4/           per-person findings
tests/                    integrity and API tests
```

## GitHub collaboration workflow

1. Clone this repository and create a virtual environment.
2. Create a feature branch from the latest `main` (see suggested names below).
3. Implement models only in your assigned `experiments/` and `models/` directories.
4. Use the shared dataset and saved development folds.
5. Write results to your own `results/personN/`, `figures/personN/`, and `docs/personN/` paths.
6. Run `pytest` and `python team_shared_setup/verify_shared_setup.py`.
7. Open a pull request into `main`. Request review for shared-code changes.
8. After all workstreams have development results, the group combines comparisons and decides when to open the test partition.

Suggested branches:

- `feature/logistic-naive-bayes`
- `feature/knn-svm`
- `feature/decision-tree-random-forest`
- `feature/pca-clustering`
- `feature/shared-<short-topic>` for reviewed shared-module changes

Do not commit directly to `main` for ongoing feature work. Do not overwrite another member’s results.

## Reproducibility

- Random seed: 245
- Dataset and split checksums are listed above and checked by `verify_shared_setup.py`
- Primary selection metric on development folds: macro-F1
- scikit-learn is used for classical models; keep dependencies aligned with `requirements.txt`
- The large tuned Random Forest binary is gitignored; metrics and configs remain tracked under `results/person3/random_forest/`
