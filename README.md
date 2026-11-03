# Kepler Exoplanet Classification (DATA 245)

Classical machine-learning classification of Kepler Objects of Interest (KOI) into CONFIRMED, CANDIDATE, and FALSE POSITIVE. Person 3 (Danish Waseem) completed Stages 1–5B for Decision Tree and Random Forest. The repository is organized so four teammates can work in parallel on separate model packages.

GitHub repository name: `kepler-exoplanet-classification`

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

Primary features: 76 leakage-aware columns in `data/processed/proposed_leakage_aware_features.csv` (still **proposed** until the team finalizes them). Primary modeling uses 7,381 development rows with a measured transit depth. The 363 incomplete KOIs stay in the split file and are excluded from the primary experiment.

## Team responsibilities

| Person | Name | Owns | Directory |
| --- | --- | --- | --- |
| 1 | Sanjana Reddy Katham | Logistic Regression, Naive Bayes, baselines, class distribution | `experiments/person1_baselines/`, `src/exoplanet/models/person1_baselines/` |
| 2 | Venkata Sai Bhavana Yarrapragada | KNN, SVM, scaling, hyperparameter tuning | `experiments/person2_knn_svm/`, `src/exoplanet/models/person2_knn_svm/` |
| 3 | Danish Waseem | Decision Tree, Random Forest, pruning, feature importance, centroid sensitivity | `experiments/person3_trees/`, `src/exoplanet/stage2.py`–`stage5b.py` |
| 4 | Nathan Pak | PCA, K-Means, dimensionality reduction, clustering visualization | `experiments/person4_pca_clustering/`, `src/exoplanet/models/person4_pca_clustering/` |

Everyone shares the dataset, feature list, splits, and evaluation helpers. See `docs/TEAM_WORKFLOW.md` and `CONTRIBUTING.md`. Do not edit another person’s directories.

## Setup (from the repository root)

```bash
git clone https://github.com/danwaseem/kepler-exoplanet-classification.git
cd kepler-exoplanet-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the dataset, split, and shared API:

```bash
python team_shared_setup/verify_shared_setup.py
pytest
```

`verify_shared_setup.py` reads the canonical files under `data/raw/` and `data/processed/`. It does not need the duplicate CSV copies that are gitignored.

Optional checksum print:

```bash
PYTHONPATH=src python - <<'PY'
from exoplanet.io import file_sha256
from exoplanet.config import RAW_CSV, DATA_PROCESSED, SAVED_SPLIT_SHA256
assert file_sha256(RAW_CSV) == "a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab"
assert file_sha256(DATA_PROCESSED / "proposed_splits.csv") == SAVED_SPLIT_SHA256
print("checksums ok")
PY
```

## Shared development API

Run scripts from the repository root so `pytest.ini` and relative paths resolve correctly.

```python
from exoplanet.development import load_development_bundle
from exoplanet.preprocess import make_preprocess_pipeline
from exoplanet.evaluation import cross_validate_estimator, write_experiment_results
from sklearn.pipeline import Pipeline

bundle = load_development_bundle()  # 7381 development rows, 76 features

def factory():
    return Pipeline([
        *make_preprocess_pipeline(scale=True).steps,  # False for trees / forests
        ("model", YourModel(...)),
    ])

summary = cross_validate_estimator(bundle, factory)
write_experiment_results("person2_knn_svm", "baseline", summary, {"model": "..."})
```

Rules: use the saved `dev_fold` values; fit preprocessing only on the training fold; do not score the test partition during development. Report grouped validation macro-F1 as the primary comparison metric.

## Run your own experiments

Add code only under your owned directories, then:

```bash
# example after you add a script
PYTHONPATH=src python experiments/person1_baselines/your_script.py
PYTHONPATH=src python experiments/person2_knn_svm/your_script.py
PYTHONPATH=src python experiments/person4_pca_clustering/your_script.py
```

Person 3 stages (already completed; do not overwrite results):

```bash
python run_stage1.py   # or experiments/person3_trees/run_stage1.py
python run_stage2.py
python run_stage3.py
python run_stage4.py
python run_stage5.py
python run_stage5b.py
```

Selected models (unchanged): pruned tree `ccp_alpha≈0.00111076`; tuned forest 100 trees, `max_depth=20`, `max_features='sqrt'`, `min_samples_leaf=2`, `class_weight='balanced'`. Development validation macro-F1: pruned tree 0.7716, tuned forest 0.8156. Summary: `docs/experimental_summary.md`.

## Repository layout

```
data/raw/                 NASA CSV (canonical)
data/processed/           features, splits, matrices
data/external/            column-page documentation
src/exoplanet/            shared loaders, preprocess, evaluation
src/exoplanet/models/     per-person model packages
experiments/              per-person entry points
results/                  stage and teammate metrics
figures/                  plots
tests/                    integrity and API tests
docs/                     findings and team workflow
```

## Contribute

1. Create a feature branch from `main` (see `CONTRIBUTING.md`).
2. Work only in your `models/` and `experiments/` directories.
3. Run `pytest` and `python team_shared_setup/verify_shared_setup.py`.
4. Open a pull request. Do not push to `main` without review.

Suggested branches: `feature/logistic-naive-bayes`, `feature/knn-svm`, `feature/decision-tree-random-forest`, `feature/pca-clustering`.

## What is not done yet

Final test-set evaluation, teammate model implementations, team finalization of the feature list, and the incomplete-KOI sensitivity experiment remain open. Do not open the test partition until the team agrees.
