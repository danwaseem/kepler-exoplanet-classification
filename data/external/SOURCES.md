# NASA documentation used for the feature audit

Retrieved 2026-10-31 for the DATA 245 Stage 1 leakage audit.

| File | Official page | Role |
| --- | --- | --- |
| `koi_cumulative_columns.html` | https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html | Column definitions for the Kepler Objects of Interest cumulative table. Page text says it was last updated 11 February 2021. |
| `kepler_tce_columns.html` | https://exoplanetarchive.ipac.caltech.edu/docs/API_tce_columns.html | Defines `tce_bin_oedp_sig`. Used only because `koi_bin_oedp_sig` is in the CSV and is not listed on the KOI column page. |
| `kepler_tce_koi_table_definitions_2012.html` | https://exoplanetarchive.ipac.caltech.edu/docs/TableColumnDescriptors.html | Kepler project table definitions delivered 7 December 2012. Describes the odd-even depth statistic in the KOI table. It does not use the name `koi_bin_oedp_sig`. |

The cumulative table itself is the file supplied with the project, `koi_cumulative_2026-10-06.csv`. DOI: 10.26133/NEA4. The Stage 1 scripts copy it into `data/raw/` and do not modify it.

`koi_columns_parsed.csv` is generated from the KOI column page by `python -m exoplanet.stage1`.
