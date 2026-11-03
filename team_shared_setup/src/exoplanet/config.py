"""Project paths, seeds, and explicit feature-policy thresholds."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_FILENAME = "koi_cumulative_2026-10-06.csv"
RAW_CSV = ROOT / "data" / "raw" / RAW_FILENAME
SOURCE_CSV = ROOT / RAW_FILENAME

DATA_EXTERNAL = ROOT / "data" / "external"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "stage1"
FIGURES = ROOT / "figures" / "eda"
DOCS = ROOT / "docs"

KOI_DICTIONARY_HTML = DATA_EXTERNAL / "koi_cumulative_columns.html"
TCE_DICTIONARY_HTML = DATA_EXTERNAL / "kepler_tce_columns.html"
KOI_2012_DEFINITIONS_HTML = DATA_EXTERNAL / "kepler_tce_koi_table_definitions_2012.html"

KOI_DICTIONARY_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html"
)
TCE_DICTIONARY_URL = "https://exoplanetarchive.ipac.caltech.edu/docs/API_tce_columns.html"
KOI_2012_DEFINITIONS_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/docs/TableColumnDescriptors.html"
)
DICTIONARY_RETRIEVED = "2026-10-31"
KOI_DICTIONARY_UPDATED = "2021-02-11"

DATASET_DOI = "10.26133/NEA4"
DATASET_SOURCE = (
    "NASA Exoplanet Archive, Kepler Objects of Interest Cumulative Table"
)
# Date embedded in the filename and stated in the project proposal.
DATASET_VERSION_DATE = "2026-10-06"

RANDOM_SEED = 245
# SHA-256 of data/processed/proposed_splits.csv after the seed-245 assignment.
# Stage 1 verifies this file and does not rewrite it.
SAVED_SPLIT_SHA256 = "d0a2892a312ddfb47ae4fe37650563ec566ee254a98c781598485992442b021a"
# One fold of a 5-fold grouped split is the held-out test set (design target 20%).
HOLDOUT_N_SPLITS = 5
HOLDOUT_FOLD_INDEX = 0
DEV_CV_N_SPLITS = 5

TARGET_COLUMN = "koi_disposition"
GROUP_COLUMN = "kepid"
ID_COLUMN = "kepoi_name"
CLASS_ORDER = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]

# A column is part of the unfitted block when it is missing for every
# fittype == "none" row and missing for less than this fraction of other rows.
UNFITTED_BLOCK_OTHER_MISSING_MAX = 0.05

# Residual gaps above this fraction are not median-imputed.
MAX_RESIDUAL_MISSING_FRACTION = 0.25

# err2 is dropped only when it matches -err1 on at least this fraction of pairs.
SYMMETRIC_ERROR_FRACTION = 0.99
MIN_PAIRS_FOR_SYMMETRY = 100
SYMMETRIC_RTOL = 1e-4
SYMMETRIC_ATOL = 1e-6

# A delivery-sized provenance group. Used to detect columns that are entirely
# absent from one KOI parameter delivery.
STRUCTURAL_GROUP_MIN_ROWS = 100
STRUCTURAL_PRESENT_MAX_MISSING = 0.02

LEAKAGE_COLUMNS = [
    "koi_pdisposition",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "kepler_name",
    "koi_comment",
]

# Metadata, identifiers, model names, and file paths. kepid is retained only
# as the grouping key and is not a model feature.
METADATA_COLUMNS = [
    "kepid",
    "kepoi_name",
    "koi_tce_plnt_num",
    "koi_vet_stat",
    "koi_vet_date",
    "koi_disp_prov",
    "koi_parm_prov",
    "koi_sparprov",
    "koi_tce_delivname",
    "koi_delivname",
    "koi_fittype",
    "koi_limbdark_mod",
    "koi_trans_mod",
    "koi_quarters",
    "koi_datalink_dvr",
    "koi_datalink_dvs",
    "ra_str",
    "dec_str",
]

# Sky coordinates, including absolute flux-weighted source positions.
# Offsets from the target are not in this set.
SKY_POSITION_COLUMNS = ["ra", "dec", "koi_fwm_sra", "koi_fwm_sdec"]

LIMB_DARKENING_COLUMNS = [
    "koi_ldm_coeff1",
    "koi_ldm_coeff2",
    "koi_ldm_coeff3",
    "koi_ldm_coeff4",
]

# The column page leaves the uncertainty cell for koi_insol blank, but the
# CSV contains these companion columns.
INSOL_ERROR_COLUMNS = {
    "koi_insol_err1": "koi_insol",
    "koi_insol_err2": "koi_insol",
}

# Present in the CSV, absent from the KOI column page. The TCE column page
# defines the matching statistic tce_bin_oedp_sig.
ODD_EVEN_COLUMN = "koi_bin_oedp_sig"

# Transit properties the column-page introduction names as free fit parameters,
# plus fitted stellar density, whose own description says it is fitted directly.
FREE_FIT_PARAMETERS = {
    "koi_period",
    "koi_time0bk",
    "koi_ror",
    "koi_dor",
    "koi_impact",
    "koi_srho",
}

PIXEL_OFFSET_COLUMNS = [
    "koi_fwm_srao",
    "koi_fwm_sdeco",
    "koi_fwm_prao",
    "koi_fwm_pdeco",
    "koi_fwm_stat_sig",
    "koi_dicco_mra",
    "koi_dicco_mdec",
    "koi_dicco_msky",
    "koi_dikco_mra",
    "koi_dikco_mdec",
    "koi_dikco_msky",
]
