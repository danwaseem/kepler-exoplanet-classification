# Stage 1 findings

These counts were produced by `python -m exoplanet.stage1`. The feature list and split are a proposal for the team. They are not final until the other members agree. This stage does not train a Decision Tree or a Random Forest.

## Dataset record

- Source: NASA Exoplanet Archive, Kepler Objects of Interest Cumulative Table
- DOI: 10.26133/NEA4
- Version date in the filename: 2026-10-06
- Local file modification time: 2026-10-13T00:33:31-07:00
- Canonical file: `data/raw/koi_cumulative_2026-10-06.csv`
- SHA-256: `a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab`
- Bytes: 11,843,781
- Shape: 9,564 rows × 153 columns
- Duplicate rows: 0
- Duplicate `kepoi_name`: 0
- Host stars (`kepid`): 8,214
- Host stars with more than one KOI: 938 (2,288 KOIs)
- Host stars with mixed dispositions: 327
- Column page: https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html
- Column page last updated: 2021-02-11; retrieved 2026-10-31
- Random seed: 245

### Disposition counts

| Disposition | Count | Share |
| --- | ---: | ---: |
| CONFIRMED | 2,748 | 28.7% |
| CANDIDATE | 1,977 | 20.7% |
| FALSE POSITIVE | 4,839 | 50.6% |

FALSE POSITIVE is the largest class. Accuracy later has to be read against that base rate. This note is not a trained baseline model.

### Values that are not 0/1 false-positive flags

These values are left unchanged in the raw file. The flags are excluded from the scientific feature set. The diagnostic matrix keeps the stored value.

- `K00477.01` has `koi_fpflag_nt` = 465.

### Repeated numeric values checked as possible sentinels

The column page notes that older deliveries sometimes used 99 or -99 for processing problems and that those codes were later stored as nulls. A stellar-temperature uncertainty of 99 K is also a plausible measurement, so these cells were not recoded.

- `koi_steff_err1` equals 99 on 23 rows.

## Rows without a transit depth

- `koi_fittype` = none: 369 rows
- Rows with missing `koi_depth`: 363
- Fit type none but depth present: 6
- Depth missing and fit type not none: 0

The column page says fit type none means a fit is not provided and only orbital period, transit epoch, and transit duration are reported. In this file, 6 of the 369 rows labeled none still have a transit depth and the associated fitted parameters, so those rows stay in the primary matrix. Every row with missing depth has fit type none. FALSE POSITIVE is 70.8% of rows with no transit depth and 49.8% of rows with a transit depth.

Median imputation is not appropriate for this block. It would fill planet and stellar parameters the archive did not measure. Leaving the cells missing is also not neutral: scikit-learn 1.9 trees can split on a missing value, and here the missingness is associated with disposition. The primary experiment therefore uses only rows with a measured transit depth. The rows without depth are not deleted. Each keeps the development or test assignment already stored in `data/processed/proposed_splits.csv`, and those assignments are also listed in `data/processed/incomplete_koi_split_assignments.csv` for a later sensitivity experiment.

### Disposition of rows with no transit depth

| Disposition | Count | Share |
| --- | ---: | ---: |
| CONFIRMED | 2 | 0.6% |
| CANDIDATE | 104 | 28.7% |
| FALSE POSITIVE | 257 | 70.8% |

### Disposition of rows with a transit depth

| Disposition | Count | Share |
| --- | ---: | ---: |
| CONFIRMED | 2,746 | 29.8% |
| CANDIDATE | 1,873 | 20.4% |
| FALSE POSITIVE | 4,582 | 49.8% |

### Columns missing on every row with no transit depth (74)

- `koi_gmag_err`
- `koi_rmag_err`
- `koi_imag_err`
- `koi_zmag_err`
- `koi_kepmag_err`
- `koi_limbdark_mod`
- `koi_ldm_coeff4`
- `koi_ldm_coeff3`
- `koi_ldm_coeff2`
- `koi_ldm_coeff1`
- `koi_trans_mod`
- `koi_model_snr`
- `koi_model_dof`
- `koi_model_chisq`
- `koi_time0bk_err1`
- `koi_time0bk_err2`
- `koi_eccen`
- `koi_eccen_err1`
- `koi_eccen_err2`
- `koi_longp`
- `koi_longp_err1`
- `koi_longp_err2`
- `koi_prad`
- `koi_prad_err1`
- `koi_prad_err2`
- `koi_sma`
- `koi_sma_err1`
- `koi_sma_err2`
- `koi_impact`
- `koi_impact_err1`
- `koi_impact_err2`
- `koi_duration_err1`
- `koi_duration_err2`
- `koi_ingress`
- `koi_ingress_err1`
- `koi_ingress_err2`
- `koi_depth`
- `koi_depth_err1`
- `koi_depth_err2`
- `koi_period_err1`
- `koi_period_err2`
- `koi_ror`
- `koi_ror_err1`
- `koi_ror_err2`
- `koi_dor`
- `koi_dor_err1`
- `koi_dor_err2`
- `koi_incl`
- `koi_incl_err1`
- `koi_incl_err2`
- `koi_teq`
- `koi_teq_err1`
- `koi_teq_err2`
- `koi_steff`
- `koi_steff_err1`
- `koi_steff_err2`
- `koi_slogg`
- `koi_slogg_err1`
- `koi_slogg_err2`
- `koi_smet`
- `koi_smet_err1`
- `koi_smet_err2`
- `koi_srad`
- `koi_srad_err1`
- `koi_srad_err2`
- `koi_smass`
- `koi_smass_err1`
- `koi_smass_err2`
- `koi_sage`
- `koi_sage_err1`
- `koi_sage_err2`
- `koi_sparprov`
- `koi_time0_err1`
- `koi_time0_err2`

### Columns observed on every row with no transit depth (27)

- `kepid`
- `kepoi_name`
- `ra`
- `ra_err`
- `ra_str`
- `dec`
- `dec_err`
- `dec_str`
- `koi_rmag`
- `koi_kepmag`
- `koi_delivname`
- `koi_vet_stat`
- `koi_disposition`
- `koi_pdisposition`
- `koi_count`
- `koi_time0bk`
- `koi_duration`
- `koi_period`
- `koi_vet_date`
- `koi_disp_prov`
- `koi_parm_prov`
- `koi_time0`
- `koi_fpflag_nt`
- `koi_fpflag_ss`
- `koi_fpflag_co`
- `koi_fpflag_ec`
- `koi_fittype`

Columns missing for every row that lacks a transit depth, and missing on less than 5% of rows that have a depth:

- `koi_ldm_coeff4`
- `koi_ldm_coeff3`
- `koi_ldm_coeff2`
- `koi_ldm_coeff1`
- `koi_model_snr`
- `koi_time0bk_err1`
- `koi_time0bk_err2`
- `koi_eccen`
- `koi_prad`
- `koi_prad_err1`
- `koi_prad_err2`
- `koi_sma`
- `koi_impact`
- `koi_impact_err1`
- `koi_impact_err2`
- `koi_duration_err1`
- `koi_duration_err2`
- `koi_depth`
- `koi_depth_err1`
- `koi_depth_err2`
- `koi_period_err1`
- `koi_period_err2`
- `koi_ror`
- `koi_ror_err1`
- `koi_ror_err2`
- `koi_dor`
- `koi_dor_err1`
- `koi_dor_err2`
- `koi_incl`
- `koi_teq`
- `koi_steff`
- `koi_steff_err1`
- `koi_steff_err2`
- `koi_slogg`
- `koi_slogg_err1`
- `koi_slogg_err2`
- `koi_smet`
- `koi_smet_err1`
- `koi_smet_err2`
- `koi_srad`
- `koi_srad_err1`
- `koi_srad_err2`
- `koi_smass`
- `koi_smass_err1`
- `koi_smass_err2`
- `koi_time0_err1`
- `koi_time0_err2`

## Feature decisions

Proposed scientific features: 76.
Excluded columns: 77.

### Included features by family

- **model_derived_transit** (7): `koi_prad`, `koi_sma`, `koi_duration`, `koi_depth`, `koi_incl`, `koi_teq`, `koi_insol`
- **multiplicity** (1): `koi_count`
- **photometry** (8): `koi_gmag`, `koi_rmag`, `koi_imag`, `koi_zmag`, `koi_jmag`, `koi_hmag`, `koi_kmag`, `koi_kepmag`
- **stellar** (5): `koi_steff`, `koi_slogg`, `koi_smet`, `koi_srad`, `koi_smass`
- **tce_statistic** (1): `koi_model_snr`
- **transit_fit** (6): `koi_time0bk`, `koi_impact`, `koi_period`, `koi_ror`, `koi_dor`, `koi_srho`
- **uncertainty** (38): `koi_jmag_err`, `koi_hmag_err`, `koi_kmag_err`, `koi_time0bk_err1`, `koi_prad_err1`, `koi_prad_err2`, `koi_impact_err1`, `koi_impact_err2`, `koi_duration_err1`, `koi_depth_err1`, `koi_period_err1`, `koi_ror_err1`, `koi_ror_err2`, `koi_dor_err1`, `koi_steff_err1`, `koi_steff_err2`, `koi_slogg_err1`, `koi_slogg_err2`, `koi_smet_err1`, `koi_smet_err2`, `koi_srad_err1`, `koi_srad_err2`, `koi_smass_err1`, `koi_smass_err2`, `koi_fwm_srao_err`, `koi_fwm_sdeco_err`, `koi_fwm_prao_err`, `koi_fwm_pdeco_err`, `koi_dicco_mra_err`, `koi_dicco_mdec_err`, `koi_dicco_msky_err`, `koi_dikco_mra_err`, `koi_dikco_mdec_err`, `koi_dikco_msky_err`, `koi_insol_err1`, `koi_insol_err2`, `koi_srho_err1`, `koi_srho_err2`
- **vetting_measurement** (10): `koi_fwm_srao`, `koi_fwm_sdeco`, `koi_fwm_prao`, `koi_fwm_pdeco`, `koi_dicco_mra`, `koi_dicco_mdec`, `koi_dicco_msky`, `koi_dikco_mra`, `koi_dikco_mdec`, `koi_dikco_msky`

### Exclusion reasons

| Reason | Columns |
| --- | ---: |
| unavailable | 24 |
| metadata_or_identifier | 18 |
| disposition_leakage | 8 |
| redundant_uncertainty | 5 |
| sky_position | 4 |
| structural_missingness | 4 |
| fixed_model_input | 4 |
| no_variation | 3 |
| duplicate_measurement | 3 |
| parent_excluded | 2 |
| target | 1 |
| vetting_summary | 1 |

### Uncertainty columns

Each uncertainty column was reviewed on its own. Empty columns, columns whose parent measurement was excluded, exact negative mirrors of `err1`, and delivery-wide gaps are excluded. The others are proposed features.

- Uncertainty columns in the CSV: 68
- Proposed for inclusion: 38
- Excluded: 30

| Column | Parent | Missing on depth-measured rows | Decision | Why excluded |
| --- | --- | ---: | --- | --- |
| `dec_err` | `dec` | 0.0% | exclude | no_variation |
| `koi_depth_err2` | `koi_depth` | 1.0% | exclude | redundant_uncertainty |
| `koi_dor_err2` | `koi_dor` | 1.0% | exclude | redundant_uncertainty |
| `koi_duration_err2` | `koi_duration` | 1.0% | exclude | redundant_uncertainty |
| `koi_eccen_err1` | `koi_eccen` | 100.0% | exclude | unavailable |
| `koi_eccen_err2` | `koi_eccen` | 100.0% | exclude | unavailable |
| `koi_fwm_sdec_err` | `koi_fwm_sdec` | 4.4% | exclude | parent_excluded |
| `koi_fwm_sra_err` | `koi_fwm_sra` | 4.4% | exclude | parent_excluded |
| `koi_gmag_err` | `koi_gmag` | 100.0% | exclude | unavailable |
| `koi_imag_err` | `koi_imag` | 100.0% | exclude | unavailable |
| `koi_incl_err1` | `koi_incl` | 100.0% | exclude | unavailable |
| `koi_incl_err2` | `koi_incl` | 100.0% | exclude | unavailable |
| `koi_ingress_err1` | `koi_ingress` | 100.0% | exclude | unavailable |
| `koi_ingress_err2` | `koi_ingress` | 100.0% | exclude | unavailable |
| `koi_kepmag_err` | `koi_kepmag` | 100.0% | exclude | unavailable |
| `koi_longp_err1` | `koi_longp` | 100.0% | exclude | unavailable |
| `koi_longp_err2` | `koi_longp` | 100.0% | exclude | unavailable |
| `koi_period_err2` | `koi_period` | 1.0% | exclude | redundant_uncertainty |
| `koi_rmag_err` | `koi_rmag` | 100.0% | exclude | unavailable |
| `koi_sage_err1` | `koi_sage` | 100.0% | exclude | unavailable |
| `koi_sage_err2` | `koi_sage` | 100.0% | exclude | unavailable |
| `koi_sma_err1` | `koi_sma` | 100.0% | exclude | unavailable |
| `koi_sma_err2` | `koi_sma` | 100.0% | exclude | unavailable |
| `koi_teq_err1` | `koi_teq` | 100.0% | exclude | unavailable |
| `koi_teq_err2` | `koi_teq` | 100.0% | exclude | unavailable |
| `koi_time0_err1` | `koi_time0` | 1.0% | exclude | duplicate_measurement |
| `koi_time0_err2` | `koi_time0` | 1.0% | exclude | duplicate_measurement |
| `koi_time0bk_err2` | `koi_time0bk` | 1.0% | exclude | redundant_uncertainty |
| `koi_zmag_err` | `koi_zmag` | 100.0% | exclude | unavailable |
| `ra_err` | `ra` | 0.0% | exclude | no_variation |
| `koi_depth_err1` | `koi_depth` | 1.0% | include |  |
| `koi_dicco_mdec_err` | `koi_dicco_mdec` | 5.4% | include |  |
| `koi_dicco_mra_err` | `koi_dicco_mra` | 5.4% | include |  |
| `koi_dicco_msky_err` | `koi_dicco_msky` | 5.4% | include |  |
| `koi_dikco_mdec_err` | `koi_dikco_mdec` | 5.1% | include |  |
| `koi_dikco_mra_err` | `koi_dikco_mra` | 5.1% | include |  |
| `koi_dikco_msky_err` | `koi_dikco_msky` | 5.1% | include |  |
| `koi_dor_err1` | `koi_dor` | 1.0% | include |  |
| `koi_duration_err1` | `koi_duration` | 1.0% | include |  |
| `koi_fwm_pdeco_err` | `koi_fwm_pdeco` | 7.8% | include |  |
| `koi_fwm_prao_err` | `koi_fwm_prao` | 7.9% | include |  |
| `koi_fwm_sdeco_err` | `koi_fwm_sdeco` | 3.9% | include |  |
| `koi_fwm_srao_err` | `koi_fwm_srao` | 3.9% | include |  |
| `koi_hmag_err` | `koi_hmag` | 1.0% | include |  |
| `koi_impact_err1` | `koi_impact` | 1.0% | include |  |
| `koi_impact_err2` | `koi_impact` | 1.0% | include |  |
| `koi_insol_err1` | `koi_insol` | 0.0% | include |  |
| `koi_insol_err2` | `koi_insol` | 0.0% | include |  |
| `koi_jmag_err` | `koi_jmag` | 0.7% | include |  |
| `koi_kmag_err` | `koi_kmag` | 2.0% | include |  |
| `koi_period_err1` | `koi_period` | 1.0% | include |  |
| `koi_prad_err1` | `koi_prad` | 0.0% | include |  |
| `koi_prad_err2` | `koi_prad` | 0.0% | include |  |
| `koi_ror_err1` | `koi_ror` | 0.0% | include |  |
| `koi_ror_err2` | `koi_ror` | 0.0% | include |  |
| `koi_slogg_err1` | `koi_slogg` | 1.1% | include |  |
| `koi_slogg_err2` | `koi_slogg` | 1.1% | include |  |
| `koi_smass_err1` | `koi_smass` | 1.1% | include |  |
| `koi_smass_err2` | `koi_smass` | 1.1% | include |  |
| `koi_smet_err1` | `koi_smet` | 0.3% | include |  |
| `koi_smet_err2` | `koi_smet` | 0.3% | include |  |
| `koi_srad_err1` | `koi_srad` | 1.1% | include |  |
| `koi_srad_err2` | `koi_srad` | 1.1% | include |  |
| `koi_srho_err1` | `koi_srho` | 0.0% | include |  |
| `koi_srho_err2` | `koi_srho` | 0.0% | include |  |
| `koi_steff_err1` | `koi_steff` | 1.1% | include |  |
| `koi_steff_err2` | `koi_steff` | 1.3% | include |  |
| `koi_time0bk_err1` | `koi_time0bk` | 1.0% | include |  |

Full sentences for each decision are in `results/shared/column_audit.csv`.

### Centroid measurements and disposition flags

The column page separates pixel measurements from the centroid-offset flag. `koi_fpflag_co` is the flag inferred from centroid measurements. The offset columns are measurements from flux-weighted centroids or PRF-fit difference images, not copies of `koi_disposition`. They stay provisional because they come from the vetting analysis and may be strong predictors. `koi_fwm_stat_sig` is not an offset. The column page defines it as a significance percent, where 100% means confidence that the transit is on the target star, so it is removed from the primary set and saved in `experimental_features_with_fwm_stat_sig.csv`. Absolute source coordinates (`koi_fwm_sra`, `koi_fwm_sdec`) are excluded as sky positions.

| Column | Decision | Provisional | Group |
| --- | --- | --- | --- |
| `koi_fpflag_nt` | exclude | False | disposition_leakage |
| `koi_fpflag_ss` | exclude | False | disposition_leakage |
| `koi_fpflag_co` | exclude | False | disposition_leakage |
| `koi_fpflag_ec` | exclude | False | disposition_leakage |
| `koi_score` | exclude | False | disposition_leakage |
| `koi_pdisposition` | exclude | False | disposition_leakage |
| `koi_fwm_sra` | exclude | False | sky_position |
| `koi_fwm_sdec` | exclude | False | sky_position |
| `koi_fwm_srao` | include | True | included |
| `koi_fwm_sdeco` | include | True | included |
| `koi_fwm_prao` | include | True | included |
| `koi_fwm_pdeco` | include | True | included |
| `koi_fwm_stat_sig` | exclude | False | vetting_summary |
| `koi_dicco_mra` | include | True | included |
| `koi_dicco_mdec` | include | True | included |
| `koi_dicco_msky` | include | True | included |
| `koi_dikco_mra` | include | True | included |
| `koi_dikco_mdec` | include | True | included |
| `koi_dikco_msky` | include | True | included |
| `koi_bin_oedp_sig` | exclude | True | structural_missingness |

## Proposed split

Seed 245. `StratifiedGroupKFold` with 5 folds, shuffle on, holdout = fold 0. Development cross-validation is a separate 5-fold grouped split on the development stars only.

- Test host-star fraction: 20.0%
- Test row fraction: 20.0%
- Host stars in both partitions: 0

| Sample | Rows | Host stars | CONFIRMED | CANDIDATE | FALSE POSITIVE | Largest shift (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_sample | 9,564 | 8,214 | 28.7% | 20.7% | 50.6% | 0.00 |
| development | 7,653 | 6,573 | 28.7% | 20.7% | 50.6% | 0.00 |
| test | 1,911 | 1,641 | 28.7% | 20.7% | 50.6% | 0.01 |
| development_primary_rows | 7,381 | 6,357 | 29.8% | 20.4% | 49.8% | 1.03 |
| test_primary_rows | 1,820 | 1,570 | 30.2% | 20.2% | 49.6% | 1.43 |

Shifts are percentage points relative to the full table of 9,564 rows.

## Preprocessing

The proposed pipeline is a median imputer with no scaler. Trees do not need standardized features. The imputer is fit on development training rows only. Stage 1 does not transform the test partition.

Rows without `koi_depth` stay in the split file and are omitted from the primary matrix, so their missing fitted parameters are not imputed.

Residual gaps are median-imputed only when they are at or below 25% of the rows that have a transit depth. Columns with a residual gap: 50.

- `koi_gmag`
- `koi_rmag`
- `koi_imag`
- `koi_zmag`
- `koi_jmag`
- `koi_jmag_err`
- `koi_hmag`
- `koi_hmag_err`
- `koi_kmag`
- `koi_kmag_err`
- `koi_kepmag`
- `koi_time0bk_err1`
- `koi_impact_err1`
- `koi_impact_err2`
- `koi_duration_err1`
- `koi_depth_err1`
- `koi_period_err1`
- `koi_dor_err1`
- `koi_incl`
- `koi_steff_err1`
- `koi_steff_err2`
- `koi_slogg_err1`
- `koi_slogg_err2`
- `koi_smet`
- `koi_smet_err1`
- `koi_smet_err2`
- `koi_srad_err1`
- `koi_srad_err2`
- `koi_smass_err1`
- `koi_smass_err2`
- `koi_fwm_srao`
- `koi_fwm_srao_err`
- `koi_fwm_sdeco`
- `koi_fwm_sdeco_err`
- `koi_fwm_prao`
- `koi_fwm_prao_err`
- `koi_fwm_pdeco`
- `koi_fwm_pdeco_err`
- `koi_dicco_mra`
- `koi_dicco_mra_err`
- `koi_dicco_mdec`
- `koi_dicco_mdec_err`
- `koi_dicco_msky`
- `koi_dicco_msky_err`
- `koi_dikco_mra`
- `koi_dikco_mra_err`
- `koi_dikco_mdec`
- `koi_dikco_mdec_err`
- `koi_dikco_msky`
- `koi_dikco_msky_err`

The broad diagnostic matrix adds `koi_pdisposition`, `koi_score`, the four `koi_fpflag` columns, and `has_kepler_name` on the same rows. It is not a scientific feature set, and its scores must not be reported as prediction performance.

## Proposal status

Pending agreement from Sanjana Reddy Katham, Nathan Pak, and Venkata Sai Bhavana Yarrapragada. Person 3 prepared this configuration so the team can share one split and one leakage-aware column list.
