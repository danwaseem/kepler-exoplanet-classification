# Person 2 findings: KNN and SVM

Venkata Sai Bhavana Yarrapragada. Development partition only: 7,381 KOIs with a measured transit depth, the saved `dev_fold` values from `data/processed/proposed_splits.csv` (split SHA-256 `d0a2892a…b021a`), grouped 5-fold cross-validation, and macro-F1 as the primary metric. Median imputation and scaling are fit inside each training fold. **The test partition was not scored.**

Configuration A is the 76-feature primary set. Configuration B drops the 20 provisional centroid-offset columns, leaving 56 features (same definitions as `docs/shared/TEAM_WORKFLOW.md`).

## Steps

1. **Scaling experiment.** KNN (k=5), RBF SVM and linear SVM were each run with no scaling, StandardScaler and MinMaxScaler, for both configurations.
2. **KNN k sweep and tuning.** Odd k from 1 to 51, then a grid search over k, distance metric (euclidean or manhattan), weighting (uniform or distance) and scaler.
3. **SVM tuning.** Linear SVM grid over C and class_weight; RBF SVM grid over C, gamma and class_weight.
4. **Final comparison.** Default and tuned models scored side by side, with per-class scores, out-of-fold ROC-AUC and confusion matrices.
5. **Larger C and class thresholds.** An extended C grid (10–300) for the RBF SVM, and per-class decision-score offsets chosen with nested folds.
6. **Bias–variance.** Training versus validation macro-F1 across k, C and gamma.

The exploration is in `experiments/person2_knn_svm/KNN_SVM_Yarrapragada.ipynb`; its tables are in `results/person2/notebook/` and its figures in `figures/person2/`. The final models were re-scored with the shared `cross_validate_estimator` helper by `experiments/person2_knn_svm/run_final_models.py`.

## Results (from `results/person2/summary.csv`)

Mean ± std of grouped validation macro-F1 across the 5 folds (std uses n−1). ROC-AUC is the one-vs-rest macro average on pooled out-of-fold scores. Candidate recall is from pooled out-of-fold predictions.

| Experiment | Config | Features | Macro-F1 | Accuracy | ROC-AUC | Candidate recall |
| --- | :-: | ---: | ---: | ---: | ---: | ---: |
| knn_default_unscaled | A | 76 | 0.6069 ± 0.0145 | 0.6574 | 0.8031 | 0.3748 |
| knn_default_scaled | A | 76 | 0.7313 ± 0.0139 | 0.7711 | 0.8939 | 0.5229 |
| knn_tuned | A | 76 | 0.7364 ± 0.0146 | 0.7811 | 0.9027 | 0.4950 |
| linear_svm_default | A | 76 | 0.7654 ± 0.0041 | 0.8149 | 0.9128 | 0.4850 |
| linear_svm_tuned | A | 76 | 0.7750 ± 0.0122 | 0.8083 | 0.9127 | 0.6047 |
| rbf_svm_default_unscaled | A | 76 | 0.2627 ± 0.0916 | 0.5148 | 0.6250 | 0.0000 |
| rbf_svm_default_scaled | A | 76 | 0.7782 ± 0.0117 | 0.8216 | 0.9132 | 0.5336 |
| **rbf_svm_tuned** | A | 76 | **0.7888 ± 0.0116** | 0.8160 | 0.9218 | 0.6757 |
| knn_tuned_no_centroid | B (retuned) | 56 | 0.6358 ± 0.0054 | 0.6964 | 0.8267 | 0.3455 |
| linear_svm_tuned_no_centroid | B (retuned) | 56 | 0.7069 ± 0.0061 | 0.7551 | 0.8747 | 0.4532 |
| **rbf_svm_tuned_no_centroid** | B (retuned) | 56 | **0.7316 ± 0.0067** | 0.7565 | 0.8873 | 0.6372 |
| knn_tuned_no_centroid_fixed_params | B (A params) | 56 | 0.6301 ± 0.0107 | 0.7086 | 0.8397 | 0.2724 |
| rbf_svm_tuned_no_centroid_fixed_params | B (A params) | 56 | 0.7296 ± 0.0086 | 0.7557 | 0.8885 | 0.6233 |

The best model is the tuned RBF SVM. Removing the centroid offsets lowers its macro-F1 by about 0.057 (0.7888 → 0.7316), and lowers tuned KNN by about 0.10. This matches the direction of the tree and forest result in `results/person3/centroid_sensitivity/`.

## Decisions

- **StandardScaler for all models.** It was best for KNN, the linear SVM and the RBF SVM. Without scaling, the RBF SVM falls to 0.26 macro-F1 and never predicts Candidate. MinMaxScaler was worse than StandardScaler in every case.
- **Final RBF SVM: C=10, gamma="scale", class_weight="balanced"** (configuration A). C=100 with gamma=0.003 scored only 0.0009 higher (0.7897 versus 0.7888). That difference is well inside the ±0.01 fold-to-fold variation, and a smaller C means stronger regularization, so C=10 is kept. Configuration B (retuned) uses C=30, gamma=0.01.
- **Other final hyperparameters.** KNN A: k=7, euclidean, distance weighting. KNN B: k=5, euclidean, uniform weighting. Linear SVM: C=10, class_weight="balanced". All are listed in `src/exoplanet/models/person2_knn_svm/models.py`.
- **Class thresholds gave no gain.** Per-class decision offsets changed RBF SVM macro-F1 from 0.7890 to 0.7887 (A) and from 0.7328 to 0.7331 (B). They were not adopted.
- **Configuration B is reported two ways:** retuned on the 56 features, and with the configuration-A hyperparameters (the approach used in `results/person3/centroid_sensitivity/`). The two agree to within 0.006 macro-F1.

## Limitations

- The tuned scores are the scores used to pick the hyperparameters. They are not nested cross-validation estimates.
- Distance-weighted KNN fits its training folds almost perfectly, so its train–validation gap (0.26) looks large even though its validation score is the best KNN result.
- The 76-feature list and the split are still a team proposal (`data/processed/proposal_status.json`).
- **The test partition was not scored.**
