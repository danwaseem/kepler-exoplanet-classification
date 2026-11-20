# Final plan approval: individual phone calls

Date: November 20, 2026
Format: Separate phone calls from Danish Waseem to each teammate

| Teammate | Call time | Approved |
| --- | --- | --- |
| Sanjana Reddy Katham | 10:56 AM | Yes |
| Venkata Sai Bhavana Yarrapragada | 11:03 AM | Yes |
| Nathan Pak | 11:08 AM | Yes |

## Decisions approved

1. Data: 76-feature list, host-star grouped split (SHA-256 d0a2892a...),
   and 7,381 eligible development rows are final. The 363-row
   sensitivity experiment is dropped as optional.
2. Configuration A (76 features) is primary. Configuration B (56
   features, no centroid columns) is reported from development folds
   only, next to A for every model.
3. Final models and settings are frozen:
   - Random Forest: 100 trees, max_depth 20, max_features sqrt,
     min_samples_leaf 2, class_weight balanced
   - RBF SVM: C=10, gamma scale, class_weight balanced
   - Linear SVM: C=10, class_weight balanced
   - Logistic Regression: L2, C=1000, class_weight balanced
   - Pruned Decision Tree: ccp_alpha=0.0011107571059714625
   - KNN: k=7, euclidean, distance weighting
   - Naive Bayes: signed-log transform
4. Macro-F1 is the primary metric. Random Forest is the expected best
   model, stated before the test.
5. The test partition is scored once, by Danish Waseem, after the
   protocol PR is merged. Results are reported as they come out.
6. Demo model: both the pruned Decision Tree and the Random Forest.
7. CRediT roles: to be confirmed by each member before the slides.
