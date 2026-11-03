# Stage 5: feature importance

The feature list is still provisional. Importance is computed on the 7,381 development rows with a measured transit depth and the saved `dev_fold` values. Each model and its median imputer are fit on that fold's training rows. Permutation importance is the drop in validation macro-F1 when a raw feature is shuffled, repeated 10 times. The test partition is not used. These results do not change the selected tree or forest.

## Sample

- Development rows: 7,381
- Development host stars: 6,357
- Features: 76
- Saved split SHA-256: `d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a`
- Pruned tree: `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `ccp_alpha=0.00111076`
- Tuned forest: 100 trees, `max_depth=20`, `max_features='sqrt'`, `min_samples_leaf=2`, `class_weight='balanced'`

The validation macro-F1 of these refits matches the Stage 3 and Stage 4 fold scores.

## Tuned forest, permutation importance

Values are the mean across the five validation folds. The standard deviation is across those five fold means, not a hypothesis test. Each fold mean averages 10 shuffles. The saved CSV also contains the mean of the within-fold repeat standard deviations.

| Rank | Column | Mean | Std across folds | Best fold rank | Worst fold rank | Provisional centroid |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `koi_model_snr` | 0.0183 | 0.0050 | 1 | 2 | False |
| 2 | `koi_dikco_msky` | 0.0150 | 0.0028 | 1 | 2 | True |
| 3 | `koi_dicco_msky` | 0.0088 | 0.0026 | 3 | 3 | True |
| 4 | `koi_prad` | 0.0024 | 0.0027 | 5 | 72 | False |
| 5 | `koi_dor` | 0.0024 | 0.0023 | 4 | 34 | False |
| 6 | `koi_duration_err1` | 0.0022 | 0.0035 | 7 | 68 | False |
| 7 | `koi_srho_err2` | 0.0022 | 0.0022 | 9 | 42 | False |
| 8 | `koi_count` | 0.0022 | 0.0028 | 6 | 61 | False |
| 9 | `koi_time0bk_err1` | 0.0021 | 0.0028 | 5 | 60 | False |
| 10 | `koi_duration` | 0.0021 | 0.0029 | 5 | 38 | False |
| 11 | `koi_dicco_msky_err` | 0.0019 | 0.0044 | 4 | 73 | True |
| 12 | `koi_ror` | 0.0018 | 0.0032 | 4 | 76 | False |
| 13 | `koi_fwm_sdeco_err` | 0.0018 | 0.0018 | 6 | 37 | True |
| 14 | `koi_fwm_srao` | 0.0017 | 0.0013 | 11 | 29 | True |
| 15 | `koi_dikco_mdec_err` | 0.0016 | 0.0034 | 4 | 69 | True |

The forest permutation ranking does not meet the pre-specified stability rule (mean pairwise Spearman at least 0.8 and at least 8 of the ten leading features with worst-fold rank 15 or better). 3 of those ten features meet the rank bound. Their worst-fold ranks are 2, 2, 3, 72, 34, 68, 42, 61, 60, 38.

Mean pairwise Spearman correlation of the five fold-level importance vectors: 0.1780 (minimum pair 0.0323).

That correlation uses all 76 features. Many have mean drops near zero, so their order changes from fold to fold. The first three do not. `koi_model_snr` and `koi_dikco_msky` stay in ranks 1–2, and `koi_dicco_msky` is rank 3 on every fold. From rank 4 downward the mean drop is about 0.002, the across-fold standard deviation is as large as the mean, and several features have a negative drop on at least one fold. Those mean ranks are not a stable ordering. `figures/person3/feature_importance/forest_rank_by_fold.png` shows the ten leading mean ranks on each fold.

## Impurity importance

Impurity importance is the mean normalized split improvement on the training folds. It is a description of the fitted forest, not a held-out score. It can favor features that offer many similar splits. Its rank correlation with the forest permutation importance is 0.6325. Impurity importance places `koi_ror` and `koi_prad` at ranks 4 and 5. Their permutation drops are much smaller than the three leading features, and their permutation ranks move across folds. Impurity importance also gives more weight to centroid uncertainty columns than the held-out permutation ranking does.

| Rank | Column | Mean | Std across folds | Best fold rank | Worst fold rank | Provisional centroid |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `koi_model_snr` | 0.0610 | 0.0031 | 1 | 1 | False |
| 2 | `koi_dikco_msky` | 0.0469 | 0.0032 | 2 | 3 | True |
| 3 | `koi_dicco_msky` | 0.0440 | 0.0025 | 2 | 3 | True |
| 4 | `koi_ror` | 0.0403 | 0.0023 | 4 | 6 | False |
| 5 | `koi_prad` | 0.0392 | 0.0033 | 4 | 5 | False |
| 6 | `koi_fwm_sdeco_err` | 0.0305 | 0.0028 | 6 | 8 | True |
| 7 | `koi_fwm_srao_err` | 0.0300 | 0.0046 | 5 | 9 | True |
| 8 | `koi_duration_err1` | 0.0277 | 0.0017 | 6 | 10 | False |
| 9 | `koi_dor` | 0.0274 | 0.0039 | 7 | 11 | False |
| 10 | `koi_time0bk_err1` | 0.0266 | 0.0018 | 8 | 11 | False |
| 11 | `koi_prad_err2` | 0.0266 | 0.0010 | 8 | 10 | False |
| 12 | `koi_dikco_msky_err` | 0.0213 | 0.0019 | 12 | 13 | True |
| 13 | `koi_prad_err1` | 0.0205 | 0.0025 | 12 | 14 | False |
| 14 | `koi_steff_err2` | 0.0173 | 0.0017 | 14 | 17 | False |
| 15 | `koi_dikco_mdec_err` | 0.0168 | 0.0012 | 14 | 18 | True |

## Pruned tree and tuned forest

The Spearman correlation between the two models' mean permutation importances is 0.5762. 6 of each model's ten leading features are shared. The absolute drops are not on the same visual scale: shuffling `koi_prad` lowers the pruned tree's validation macro-F1 by 0.1382, and the same shuffle lowers the tuned forest's by 0.0024. The tree concentrates its splits. The forest can still use `koi_ror` and `koi_depth` when `koi_prad` is shuffled. Those three columns are strongly associated on the development rows.

### Pruned tree

| Rank | Column | Mean | Std across folds | Best fold rank | Worst fold rank | Provisional centroid |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `koi_prad` | 0.1382 | 0.0339 | 1 | 2 | False |
| 2 | `koi_model_snr` | 0.0969 | 0.0205 | 1 | 3 | False |
| 3 | `koi_dikco_msky` | 0.0765 | 0.0361 | 2 | 6 | True |
| 4 | `koi_dicco_msky` | 0.0581 | 0.0166 | 3 | 6 | True |
| 5 | `koi_dor` | 0.0571 | 0.0158 | 3 | 6 | False |
| 6 | `koi_dicco_mdec_err` | 0.0463 | 0.0263 | 4 | 36 | True |
| 7 | `koi_ror` | 0.0181 | 0.0205 | 7 | 64 | False |
| 8 | `koi_smet_err2` | 0.0099 | 0.0221 | 4 | 59 | False |
| 9 | `koi_time0bk_err1` | 0.0084 | 0.0125 | 8 | 55 | False |
| 10 | `koi_dicco_msky_err` | 0.0078 | 0.0181 | 7 | 74 | True |
| 11 | `koi_count` | 0.0062 | 0.0049 | 8 | 53 | False |
| 12 | `koi_dikco_mdec_err` | 0.0042 | 0.0061 | 9 | 39 | True |
| 13 | `koi_sma` | 0.0039 | 0.0045 | 7 | 58 | False |
| 14 | `koi_impact` | 0.0033 | 0.0037 | 8 | 59 | False |
| 15 | `koi_dikco_msky_err` | 0.0031 | 0.0046 | 12 | 41 | True |

The pruned tree permutation ranking does not meet the pre-specified stability rule (mean pairwise Spearman at least 0.8 and at least 8 of the ten leading features with worst-fold rank 15 or better). 5 of those ten features meet the rank bound. Their worst-fold ranks are 2, 3, 6, 6, 6, 36, 64, 59, 55, 74.

Mean pairwise Spearman correlation of the five pruned-tree fold-level importance vectors: 0.4206 (minimum pair 0.1069).

## Provisional centroid offsets

The pre-specified rule flags centroid dominance if the leading feature is a centroid offset, if at least 5 of the top 10 are, or if those columns hold at least half of the positive mean permutation importance.

For the tuned forest, the leading feature is `koi_model_snr`, which is not a centroid offset. Two of the top 10 are centroid offsets, and their share of the positive mean permutation importance is 0.4497. The rule is not met. The two sky-plane offsets are still ranks 2 and 3 on every forest fold, and their Spearman correlation on the development rows is 0.8603. That is a methodological concern. These columns are provisional vetting measurements. A model can separate dispositions using the centroid analysis even though the disposition label and the centroid-offset flag were held out. The selected forest was not changed because of that rank.

For the pruned tree, the leading feature is `koi_prad`. Four of the top 10 are centroid offsets, and their positive-importance share is 0.3606. The rule is not met. `koi_dikco_msky` and `koi_dicco_msky` are still inside the tree's top five on every fold. The same concern applies, and the selected tree was not changed.

## Correlated predictors

Spearman correlations are computed on the 7,381 development rows, without using the test partition and without filling missing values first. 166 feature pairs have absolute correlation at least 0.7. 60 of them involve a feature in the forest's top 15. The table shows the 20 strongest of those 60. Correlated features can share or split a permutation drop: shuffling one leaves the other intact, so the measured drop can understate a redundant signal or move between the pair. A large importance does not establish that the column causes the disposition.

`koi_dicco_msky` and `koi_dikco_msky` have Spearman correlation 0.8603, so the second and third forest features are not separate pieces of evidence. Other strong development-row associations include `koi_depth` with `koi_ror` (0.9379), `koi_prad` with `koi_ror` (0.8840), `koi_prad` with `koi_depth` (0.8063), `koi_period` with `koi_sma` (0.9966), `koi_teq` with `koi_insol` (1.0000), and `koi_slogg` with `koi_srad` (−0.9796). The last three pairs are not in the forest's top 15.

| Feature A | Feature B | Spearman | Involves a top-15 feature |
| --- | --- | ---: | --- |
| `koi_fwm_srao_err` | `koi_fwm_sdeco_err` | 0.9885 | True |
| `koi_dikco_mdec_err` | `koi_dikco_msky_err` | 0.9602 | True |
| `koi_dicco_mdec_err` | `koi_dicco_msky_err` | 0.9559 | True |
| `koi_dicco_mdec_err` | `koi_dikco_mdec_err` | 0.9523 | True |
| `koi_dicco_mra_err` | `koi_dicco_msky_err` | 0.9520 | True |
| `koi_dicco_msky_err` | `koi_dikco_msky_err` | 0.9458 | True |
| `koi_depth` | `koi_ror` | 0.9379 | True |
| `koi_time0bk_err1` | `koi_duration_err1` | 0.9363 | True |
| `koi_dicco_msky_err` | `koi_dikco_mdec_err` | 0.9141 | True |
| `koi_dicco_msky_err` | `koi_dikco_mra_err` | 0.9104 | True |
| `koi_dikco_mra_err` | `koi_dikco_mdec_err` | 0.8995 | True |
| `koi_period` | `koi_dor` | 0.8880 | True |
| `koi_prad` | `koi_ror` | 0.8840 | True |
| `koi_sma` | `koi_dor` | 0.8761 | True |
| `koi_model_snr` | `koi_fwm_srao_err` | -0.8717 | True |
| `koi_model_snr` | `koi_fwm_sdeco_err` | -0.8698 | True |
| `koi_dicco_mra_err` | `koi_dikco_mdec_err` | 0.8664 | True |
| `koi_prad` | `koi_prad_err2` | -0.8639 | True |
| `koi_dor` | `koi_teq` | -0.8604 | True |
| `koi_dor` | `koi_insol` | -0.8604 | True |

## Column-page definitions for the leading forest features

The text below is the parsed NASA column-page label and description. It is not an additional physical conclusion.

### `koi_model_snr`

Column-page label: Transit Signal-to-Noise.

Transit depth normalized by the mean uncertainty in the flux during the transits.

Mean permutation importance 0.0183 (std across folds 0.0050). Fold ranks range from 1 to 2.

### `koi_dikco_msky`

Column-page label: PRF ΔθSQ(KIC) units: arcseconds.

The angular offset in the plane of the sky between the best-fit PRF centroids from the difference image and the Kepler Input Catalog position by averaging the weighted single-quarter measurements. The KIC position is subtracted from the difference image centroids.

Mean permutation importance 0.0150 (std across folds 0.0028). Fold ranks range from 1 to 2.

### `koi_dicco_msky`

Column-page label: PRF ΔθSQ(OOT) units: arcseconds.

The angular offset on the plane of the sky between the best-fit PRF centroids from the Out-Of-Transit image and the Difference Image by averaging the weighted single-quarter measurements. The out-of-transit centroids are subtracted from the difference image centroids.

Mean permutation importance 0.0088 (std across folds 0.0026). Fold ranks range from 3 to 3.

### `koi_prad`

Column-page label: Planetary Radius (Earth radii).

The radius of the planet. Planetary radius is the product of the planet star radius ratio and the stellar radius.

Mean permutation importance 0.0024 (std across folds 0.0027). Fold ranks range from 5 to 72.

### `koi_dor`

Column-page label: Planet-Star Distance over Star Radius.

The distance between the planet and the star at mid-transit divided by the stellar radius. For the case of zero orbital eccentricity, the distance at mid-transit is the semi-major axis of the planetary orbit.

Mean permutation importance 0.0024 (std across folds 0.0023). Fold ranks range from 4 to 34.

### `koi_duration_err1`

Column-page label: Uncertainty on Transit Duration (hours).

Named on the archive column page as an uncertainty column for koi_duration. The duration of the observed transits. Duration is measured from first contact between the planet and star until last contact. Contact times are typically computed from a best-fit model produced by a Mandel-Agol (2002) model fit to a multi-quarter Kepler light curve, assuming a linear orbital ephemeris.

Mean permutation importance 0.0022 (std across folds 0.0035). Fold ranks range from 7 to 68.

### `koi_srho_err2`

Column-page label: Uncertainty on Fitted Stellar Density [g/cm*3].

Named on the archive column page as an uncertainty column for koi_srho. Fitted stellar density is a direct observable from the light curve that, in the small-planet approximation, depends only on the transit's period, depth, and duration (see Seager and Mallen-Ornelas 2003). This quantity is directly fitted in the LS and MCMC methods, and is completely independent from the listed stellar mass and radius, which are derived using ground-based photometry, spectroscopy, and other observations.

Mean permutation importance 0.0022 (std across folds 0.0022). Fold ranks range from 9 to 42.

### `koi_count`

Column-page label: Number of Planets.

Number of planet candidates identified in a system.

On every host star in this file, Stage 1 found that the column equals the number of KOI rows, including false positives. The column-page wording above is the archive definition. The measured permutation rank of this column is not stable.

Mean permutation importance 0.0022 (std across folds 0.0028). Fold ranks range from 6 to 61.

### `koi_time0bk_err1`

Column-page label: Uncertainty on Transit Epoch (BJD - 2,454,833.0).

Named on the archive column page as an uncertainty column for koi_time0bk. The time corresponding to the center of the first detected transit in Barycentric Julian Day (BJD) minus a constant offset of 2,454,833.0 days. The offset corresponds to 12:00 on Jan 1, 2009 UTC.

Mean permutation importance 0.0021 (std across folds 0.0028). Fold ranks range from 5 to 60.

### `koi_duration`

Column-page label: Transit Duration (hours).

The duration of the observed transits. Duration is measured from first contact between the planet and star until last contact. Contact times are typically computed from a best-fit model produced by a Mandel-Agol (2002) model fit to a multi-quarter Kepler light curve, assuming a linear orbital ephemeris.

Mean permutation importance 0.0021 (std across folds 0.0029). Fold ranks range from 5 to 38.


## Limitations

Permutation importance depends on the fitted model and on the other columns that remain available. Impurity importance answers a different question and is reported only as a comparison. Mean ranks below the three leading forest features are too variable to support a feature ranking. The centroid-offset columns are still provisional, and the two sky-plane offsets remain a methodological concern because they occupy stable second and third place. The selected hyperparameters were not revised from these ranks.

## Not done in this stage

The test partition was not scored and was not used to compute importance. No feature was added or removed. The pruned tree and the tuned forest were not replaced.
