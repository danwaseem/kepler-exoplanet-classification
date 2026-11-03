# DATA 245 shared setup: Exoplanet Classification

This folder is the shared data and preprocessing package for the NASA Kepler KOI cumulative table. The 76-feature list and the host-star split are a **proposal**. They are not final until Sanjana Reddy Katham, Nathan Pak, Venkata Sai Bhavana Yarrapragada, and Danish Waseem agree.

The package does not contain Decision Tree or Random Forest training code, saved models, or model scores. It does not contain test-set metrics. Do not score the test partition during model development.

## What is included

When this folder sits inside the main `kepler-exoplanet-classification` repository, verification uses the **canonical** files at the repository root:

- `../data/raw/koi_cumulative_2026-10-06.csv`
- `../data/processed/proposed_splits.csv`
- `../data/processed/proposed_leakage_aware_features.csv`

Local duplicate NASA CSV copies under `team_shared_setup/` are optional and are gitignored. Do not regenerate the split.

| Path | Role |
| --- | --- |
| `data/processed/proposed_splits.csv` | Saved development/test assignment and development CV folds (package copy) |
| `data/processed/proposed_leakage_aware_features.csv` | Proposed 76 scientific features (package copy) |
| `data/processed/column_audit.csv` | Column-by-column leakage and inclusion audit |
| `data/processed/incomplete_koi_split_assignments.csv` | The 363 incomplete KOIs and their saved assignments |
| `docs/STAGE1_DECISIONS.md` | Why those choices were made (package copy; canonical: `../docs/shared/`) |
| `docs/stage1_findings.md` | Counts produced from this table (package copy; canonical: `../docs/shared/`) |
| `src/exoplanet/` | Loader, saved-split checks, and training-fold imputation |
| `requirements.txt` | pandas, numpy, and scikit-learn |
| `verify_shared_setup.py` | Read-only checksum, feature, and split checks |

Canonical NASA CSV SHA-256: `a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab`.  
Split SHA-256: `d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a`. Do not replace either file.

## Setup

From the **repository root** (recommended after clone):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python team_shared_setup/verify_shared_setup.py
```

`verify_shared_setup.py` only reads files. It checks the NASA checksum, that all 76 features exist, that the saved split and development folds 0–4 are intact, and that no host star is in both partitions.

## How to load the shared experiment

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

import pandas as pd
from exoplanet.config import DATA_PROCESSED, RAW_CSV
from exoplanet.io import load_koi
from exoplanet.preprocess import (
    fit_imputer,
    primary_learning_frame,
    training_fold_index,
)

frame = load_koi(RAW_CSV).reset_index(drop=True)
splits = pd.read_csv(DATA_PROCESSED / "proposed_splits.csv")
features = pd.read_csv(DATA_PROCESSED / "proposed_leakage_aware_features.csv")
feature_columns = features["column"].tolist()

primary = primary_learning_frame(frame, splits, feature_columns)
development = primary.loc[primary["partition"].eq("development")].copy()
```

`primary` contains the 9,201 KOIs with a measured `koi_depth`. Use `development` for training, tuning, and model comparison. The `test` rows are held out.

### Saved folds

Seed 245. Every KOI from the same `kepid` is in one partition. Inside the development partition, `dev_fold` is 0, 1, 2, 3, or 4. Test rows have `dev_fold` -1 and are not a cross-validation fold.

Use these saved folds. Do not call `assign_splits`, and do not build a new train/test split.

For fold `k`, training rows are the development rows whose `dev_fold` is not `k`. Validation rows are the development rows whose `dev_fold` is `k`.

```python
train_index = training_fold_index(splits, fold_id=0)
numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
imputer = fit_imputer(numeric, splits["partition"], train_index)
```

`fit_imputer` refuses any index outside the development partition. Fit a new imputer inside every training fold. Do not fit it on the whole table, on all development rows before cross-validation, or on the test partition.

## The 363 incomplete KOIs

Rows with `primary_supervised_row` false have no measured transit depth. There are 363 of them. They stay in `proposed_splits.csv` with the development or test assignment and development fold they already have. `incomplete_koi_split_assignments.csv` lists those rows for a later sensitivity experiment.

The primary complete-measurement experiment does not use them. Do not median-impute the unmeasured planet and stellar parameters on those rows into the shared feature matrix.

## Provisional centroid measurements

The offset columns with `provisional` true in the feature list are flux-weighted centroid shifts and PRF-fit difference-image offsets, plus their uncertainty columns. They are measurements from the pixel files, not copies of `koi_disposition`. They can still be strong predictors because they were built for vetting. The team has not agreed to keep them.

`koi_fpflag_co`, the centroid-offset flag, is excluded. `koi_fwm_stat_sig`, the flux-weighted offset significance percent, is also excluded from the 76-column list. Do not add either column to the primary experiment unless the team agrees.

## Standardization

The shared pipeline median-imputes residual gaps and does not standardize. Fit any additional transformer inside the same training fold.

KNN, SVM, logistic regression, PCA, and K-Means may require standardization. Put `StandardScaler` after the median imputer in a scikit-learn `Pipeline`, and fit that pipeline on the training fold only:

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

scaled = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
)
scaled.fit(x_train)
```

Decision Trees and Random Forests do not need standardized inputs. Use the median imputer without a scaler for those models.

Gaussian Naive Bayes is also sensitive to feature scale. If logistic regression is standardized, fit the same within-fold scaler for Naive Bayes rather than scaling the full development matrix first.

## What each person should do with this package

Everyone uses this table, this 76-feature list, these host-star partitions, and these development folds.

- Sanjana Reddy Katham (logistic regression, Naive Bayes): fit imputation and scaling inside each training fold. Keep the test partition out of model selection.
- Venkata Sai Bhavana Yarrapragada (KNN, SVM, scaling, cross-validation, tuning): use the saved `dev_fold` values for grouped cross-validation. Standardize inside the training fold. Do not create a new split.
- Nathan Pak (PCA, K-Means): standardize inside the training fold before PCA or K-Means. Do not fit either method on the test partition.
- Danish Waseem (Decision Tree, Random Forest): do not standardize. The individual model code is outside this package.

Compare models on the development folds. Report test-set metrics only after the team agrees to open the test partition.
