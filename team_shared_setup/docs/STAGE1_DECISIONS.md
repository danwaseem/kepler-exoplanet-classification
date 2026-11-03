# Stage 1 decisions

This note explains the rules behind `docs/stage1_findings.md`. The counts in that file are written by `python scripts/run_stage1.py`. The configuration is proposed for the team and is not final.

## Source files

The modeling table is the NASA Exoplanet Archive KOI cumulative file supplied with the project, version date 2026-10-06, DOI 10.26133/NEA4. The original workspace file and `data/raw/koi_cumulative_2026-10-06.csv` have the same SHA-256:

`a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab`

Column definitions come from the archive page [Data Columns in Kepler Objects of Interest Table](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html), saved on 2026-10-31. The page itself says it was last updated 11 February 2021. `koi_bin_oedp_sig` is in the CSV and is not on that page. Its description is taken from `tce_bin_oedp_sig` on the TCE column page, which says the KOI table reports the same odd-even depth comparison. The 2012 Kepler table definitions describe that statistic and do not use the `koi_bin_oedp_sig` name.

## Rows with no transit depth

The column page says fit type `none` means a fit is not provided and only period, epoch, and duration are reported. In this file, 369 rows have that label and 363 rows are missing `koi_depth`. All 363 missing-depth rows are labeled `none`. The other 6 labeled `none` still have depth, planet radius, stellar parameters, and signal-to-noise, so they stay in the modeling matrix.

On the 363 rows, period, duration, and the BK epoch are present. Planet radius, transit depth, impact parameter, stellar parameters, and the uncertainties on period and duration are not. FALSE POSITIVE is 257 of those 363 rows (70.8%), compared with 49.8% of rows that have a depth. Two of the 363 are CONFIRMED and 104 are CANDIDATE.

Median imputation is not used for this block. It would invent parameters the archive did not measure. Leaving the cells empty is also not neutral: scikit-learn 1.9 trees can send missing values down their own branch, and this missingness is associated with disposition. The primary experiment therefore uses the complete-measurement rows only. The 363 incomplete KOIs are not permanently discarded. Each keeps the development or test assignment in `data/processed/proposed_splits.csv`, and those assignments are copied to `data/processed/incomplete_koi_split_assignments.csv` for a later sensitivity experiment. The holdout does not depend on this filter.

## What is excluded from the scientific features

- `koi_disposition` is the target.
- `koi_pdisposition`, `koi_score`, the four `koi_fpflag` columns, `kepler_name`, and `koi_comment` are disposition or vetting products. The score is defined as the fraction of Robovetter iterations that call the object a candidate. A Kepler name is defined as the name of a confirmed or validated planet. The comment is defined as the reason for a false-positive disposition.
- `kepid` is the grouping key only. `kepoi_name` and the TCE planet number are identifiers.
- Provenance, vetting status, fit type, model names, quarter bit strings, and data-validation links are metadata. Fit type is used to describe the missing block and is not a feature.
- `ra`, `dec`, and the absolute flux-weighted source coordinates `koi_fwm_sra` and `koi_fwm_sdec` are sky positions. `ra_err` and `dec_err` are 0 for every row.
- Limb-darkening coefficients are fixed inputs derived from the stellar parameters. The column page says they stay fixed during the fit.
- `koi_eccen` is 0 wherever it is filled.
- `koi_time0` is the BK epoch plus 2,454,833 days. The largest departure from that offset is 0.0005 days, and its uncertainties are identical to the BK uncertainties. `koi_time0bk` is kept.
- Where `err2` equals the negative of `err1` on at least 99% of paired rows, only `err1` is kept. Asymmetric pairs, including both insolation uncertainties and both stellar-parameter uncertainties, are kept. The insolation uncertainty cell on the column page is blank; the CSV columns were still reviewed, and both are populated on the depth-measured rows.
- Empty uncertainty columns, including equilibrium-temperature errors, semi-major-axis errors, and several magnitude errors, are excluded because this file does not contain them.
- `koi_num_transits`, `koi_max_sngle_ev`, `koi_max_mult_ev`, and `koi_bin_oedp_sig` are missing for an entire parameter delivery and present for another. Imputing them would copy one delivery's typical value onto another delivery. They are not excluded because they are the disposition. The odd-even statistic remains a light-curve vetting measurement that could be studied later on the delivery that contains it.

`koi_count` is described on the column page as the number of planet candidates. On every host star in this file it equals the number of KOI rows, including false positives, so it is proposed as system multiplicity rather than as a count of the target class. It is constant within a star.

## Centroid measurements

The column page's pixel section describes flux-weighted centroid shifts and PRF-fit difference-image offsets as measurements from target pixel files. `koi_fpflag_co` is a separate flag: the source is inferred to be a nearby star from those measurements. The offset measurements stay in the primary feature set as provisional candidates. They are not copies of `koi_disposition`. They can still be strong predictors because they were built for vetting, and the team has not agreed to this list.

`koi_fwm_stat_sig` is removed from the primary leakage-aware set. The column page defines it as a significance percent, where 100% means there is no significant flux-weighted offset and there is confidence that the transit is on the target star. That is a derived vetting summary. The same column is saved in `data/processed/experimental_features_with_fwm_stat_sig.csv` so a later run can test whether adding it changes performance. Stage 2 does not train on that experimental list.

## Split and preprocessing

Seed 245. One fold of a shuffled 5-fold `StratifiedGroupKFold` on `kepid` is the test holdout. A new 5-fold grouped split is stored for tuning and uses only development stars. No host star is in both partitions. The test set is 1,641 of 8,214 stars (20.0%) and 1,911 of 9,564 rows (20.0%). Class shares in the full development and test partitions differ from the full table by at most 0.01 percentage points.

After the depth filter, the primary development set has 7,381 rows and the primary test set has 1,820 rows. Relative to the full table, CONFIRMED is about 1.0 and 1.4 percentage points higher in those two subsets because the dropped rows are mostly false positives.

The pipeline is a median imputer and does not standardize. It is fit on development training rows only. Stage 1 does not transform the test matrix. Residual gaps on included columns, mostly photometry and centroid quantities below the 25% missing limit, are the only cells that imputer fills.

## Not established

Stage 1 does not fit a Random Forest, and it does not score the test partition. The unrestricted Decision Tree is a separate stage and uses only the development partition. Nothing here answers whether a forest reduces overfitting or which features will rank highest under permutation importance. The boxplots and the Spearman heatmap are descriptive. Period and semi-major axis move together, equilibrium temperature and insolation move together and opposite period, and stellar surface gravity moves opposite stellar radius. Later importance ranks have to be read with those dependencies.

The intermediate report's literature review of at least 13 papers is not part of this stage and has not been written.
