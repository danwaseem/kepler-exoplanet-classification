# Person 4 findings: PCA and K-Means (development partition)

**Owner:** Nathan Pak · **Status:** development-stage, not a final team result · **Test partition scored:** no

Notebook: `experiments/person4_pca_clustering/04_pca_clustering.ipynb`
Helper: `src/exoplanet/models/person4_pca_clustering/transforms.py`
Metrics: `results/person4/` · Figures: `figures/person4/`

## Setup

- Shared development bundle: 7,381 KOIs with a measured transit depth, 6,357 host stars, the proposed 76 leakage-aware features, seed 245.
- Preprocessing: shared median imputer + `StandardScaler`. The main variant adds a signed-log step, `sign(x)·log1p(|x|)`, on columns with |skew| > 3 (53 of 76 columns). The columns are chosen from the rows the pipeline is fit on.
- Why the signed-log step: with plain standardization, K-Means puts single extreme KOIs into their own clusters (smallest cluster = 1 KOI for k ≥ 4), and PCA needs more components for the same variance.
- The exploratory PCA and K-Means are fit on all development rows. They use no labels and no test rows. Every supervised score refits the preprocessing and PCA inside each saved training fold (`dev_fold`).

## PCA

| Preprocessing | PC1 | PC2 | PCs for 80% | PCs for 90% | PCs for 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| standardized | 12.8% | 8.7% | 21 | 31 | 38 |
| signed-log + standardized | 16.4% | 13.4% | 16 | **25** | **33** |

- The variance is spread out. PC1 and PC2 together explain only 29.8%.
- **PC1:** stellar size and irradiation (insolation, equilibrium temperature, stellar radius, and their uncertainties).
- **PC2:** how bright the target star is versus how uncertain the centroid offsets are. Faint stars have larger centroid errors.
- **PC3:** orbit and planet size (period, semi-major axis, planet radius, a/R*).
- In the 2D projection, the three dispositions overlap heavily. CONFIRMED KOIs sit toward low PC2, FALSE POSITIVES spread widely and fill the lower-left region, and CANDIDATES lean toward higher PC1 and PC2. The two components don't separate the classes on their own.

## Original vs PCA-reduced features (novelty experiment)

Probe: Logistic Regression (`class_weight="balanced"`), 5 saved grouped folds, macro-F1 as the primary metric.

| Inputs | Validation macro-F1 (mean ± sd) | Validation accuracy |
| --- | ---: | ---: |
| original 76 features | **0.773 ± 0.011** | 0.795 |
| PCA 2 | 0.570 ± 0.020 | 0.578 |
| PCA 5 | 0.663 ± 0.014 | 0.677 |
| PCA 10 | 0.683 ± 0.014 | 0.699 |
| PCA 15 | 0.754 ± 0.018 | 0.774 |
| PCA 20 | 0.762 ± 0.013 | 0.782 |
| PCA 25 (90% var.) | **0.766 ± 0.013** | 0.787 |
| PCA 33 (95% var.) | 0.766 ± 0.013 | 0.787 |

- 25 components (a third of the inputs) keep almost all of the linear probe's performance: macro-F1 drops by 0.007, which is smaller than one fold standard deviation.
- Below about 15 components, performance falls steeply. Class information is spread over many directions, not just the top two or three.
- The train/validation gap is small for every setting (about 0.00–0.015), so PCA is not fixing an overfitting problem for this linear model.
- This probe measures how much information PCA keeps. It does not replace Person 1's Logistic Regression results.

## K-Means

- Silhouette scores are low for every k from 2 to 10 (0.09–0.15 with the signed-log step), so the feature space has weak cluster structure.
- With k = 3 (signed-log): ARI = **0.036** and NMI = **0.088** against the dispositions. Purity is 0.498, the same as always predicting FALSE POSITIVE (49.8%).
- Agreement with the labels peaks at k = 5 (ARI 0.172), which is still low.
- The clusters are stable. Refit on each training fold and assigning the held-out fold, they match the full fit with mean ARI **0.94** (range 0.89–1.00).

| Cluster | n | CONFIRMED | CANDIDATE | FALSE POS. | Median period (d) | Median radius (R⊕) | Median SNR | Median T_eq (K) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1,147 | 4% | 45% | 51% | 279 | 2.1 | 9.4 | 314 |
| 1 | 1,644 | 15% | 13% | 72% | 4.4 | 14.9 | 44 | 1,556 |
| 2 | 4,590 | 41% | 17% | 42% | 7.3 | 2.2 | 27 | 878 |

**Interpretation:** K-Means recovers real, repeatable astrophysical groups, but not the disposition classes.

- **Cluster 0:** long-period, low-SNR signals, mostly still CANDIDATE or FALSE POSITIVE. These are hard to confirm.
- **Cluster 1:** short-period, very large "planet" radii (median about 15 R⊕) around hot, larger stars, 72% FALSE POSITIVE. This is consistent with eclipsing binaries.
- **Cluster 2:** short-period, small-radius signals, where most CONFIRMED planets are.

This answers the research question "does unsupervised clustering naturally separate types of Kepler objects?" The clusters reflect signal regimes rather than vetting outcomes.

## Limitations and next steps

- The 76-feature list is still *proposed*. If the team changes it, rerun the notebook.
- The probe uses only a linear model. Next, run the PCA inputs through KNN/SVM (Person 2) and Random Forest (Person 3) for the final comparison table.
- Optional: rerun on the 56-feature set without centroid offsets, because PC2 is driven by centroid-offset uncertainties.
- Try Gaussian Mixture or k = 5 as a sensitivity check. Report k = 3 as the primary result.
