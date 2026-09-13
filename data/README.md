# Data

**No raw data is committed to this repository.** Every source below is publicly
available, but two of them cannot be redistributed and one is address-level, so
the repo ships the code that fetches and assembles the data rather than the data
itself. The scripts in `src/` are numbered in run order and will rebuild the
tract-level feature matrix from scratch.

| Source | Used for | How to obtain |
|---|---|---|
| **ACS 5-year estimates** (Census API) | Income, rent, home value, education, tenure, vacancy, age of housing stock, race/ethnicity shares, poverty, rent burden, population | Free API key at [api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html). Run `src/01_census_pull.py` with `CENSUS_API_KEY` set. |
| **LEHD LODES (WAC)** | Job counts by industry per tract; arts / food / "gentrify" job densities | Public download, no key. `src/02_lodes_multiyear_pull.py` and `src/03_lodes_2023_pull.py` fetch `ca_wac_*` files from [lehd.ces.census.gov](https://lehd.ces.census.gov/data/). |
| **Building permits** — San Francisco, Oakland, San Jose | Residential / commercial / industrial / civic permit counts per tract-year | SF and Oakland publish permit extracts through their open-data portals; San Jose publishes PDF construction reports, parsed by `src/05_scrape_sj_construction.py` and `src/06_parse_sj_construction.py`. Records are **address-level**, so neither the raw extracts nor the geocode cache are committed. |
| **Zillow ZHVI** (Home Value Index) | **Label construction only — never a model feature** | Download from [zillow.com/research/data](https://www.zillow.com/research/data/). Zillow's terms of use do not permit redistribution, so the file is not included here. |
| **TIGER/Line census tracts** | Tract geometry for the choropleth | [census.gov/geographies/mapping-files](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) — Bay Area county tract shapefiles. |

## A note on Zillow

Zillow's ZHVI is used to compute the gentrification **label** (whether a tract's
home-value appreciation landed in the top quartile of its city). It is
deliberately excluded from the feature set: `jan_housing_value_begin` and
`jan_housing_value_end` measure housing appreciation over the same window the
label is defined on, so including them as predictors would be circular. The
exclusion is enforced in code — see the `ZILLOW_COLS` assertion in
`src/13_model_comparison.py` and the comment in `PIVOT_FEATS` in
`notebooks/01_models_lasso_smote_shap.ipynb`.

## What *is* committed

`outputs/` holds the two small derived files that are our own model output:

- `gentrification_risk_scores.csv` — 341 tracts, per-model probabilities, the
  ensemble risk score, and the top SHAP driver for each tract
- `gentrification_shap_values.csv` — per-tract SHAP values for the selected features

Both are tract-level aggregates. Nothing address-level or individual-level
appears anywhere in this repository.
