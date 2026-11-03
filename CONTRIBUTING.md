# Contributing

## Branches

Create a feature branch from the latest `main`:

| Work | Branch |
| --- | --- |
| Logistic Regression / Naive Bayes | `feature/logistic-naive-bayes` |
| KNN / SVM | `feature/knn-svm` |
| Decision Tree / Random Forest | `feature/decision-tree-random-forest` |
| PCA / K-Means | `feature/pca-clustering` |
| Shared preprocessing or evaluation | `feature/shared-<short-topic>` |

Do not commit directly to `main` once the remote repository is shared.

## Commits

- Commit only files you own, plus small shared documentation when needed.
- Prefer clear messages: `Add logistic regression grouped CV metrics`.
- Do not commit `.venv/`, caches, or regenerable ZIP archives.
- Do not rewrite `data/processed/proposed_splits.csv` or the NASA CSV.
- Do not score or commit final test-set metrics until the team agrees to open the test partition.

## Pull requests

1. Push your feature branch.
2. Open a pull request into `main`.
3. Summarize models, hyperparameters, and development validation macro-F1.
4. Link the result directory under `results/<owner>/`.
5. Request review from at least one teammate for shared-code changes.

## Shared-code policy

Files under `src/exoplanet/` outside your `models/person*/` package are shared. Change them only with a reviewed pull request. Do not edit another person's `models/` or `experiments/` directory.

## Evaluation consistency

On the development folds, report at least:

- mean ± std grouped validation accuracy
- mean ± std grouped validation macro-F1
- training macro-F1 and the train–validation gap
- per-class precision, recall, and F1 from pooled out-of-fold predictions
- the exact hyperparameters and whether features were standardized
- `split_sha256` matching `d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a`

Use `exoplanet.evaluation.cross_validate_estimator` and `write_experiment_results` when possible.
