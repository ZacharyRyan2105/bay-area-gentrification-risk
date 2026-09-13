"""
build_all_datasets.py
----------------------
Builds four labeled datasets supporting two parallel model comparisons:

  MODEL A — 1-Year Appreciation (high data volume, noisier signal)
    training_1yr.csv      6 vintages × ~400 tracts  (~2,600 rows)
    validation_1yr.csv    2020 vintage → Jan 2020→2021 appreciation
    Note: 2019→2020 vintage dropped (COVID onset — exogenous shock not in features)

  MODEL B — 5-Year Appreciation (lower volume, cleaner gentrification signal)
    training_5yr.csv      2 vintages × ~400 tracts  (~850 rows)
    validation_5yr.csv    2020 vintage → Jan 2020→2025 appreciation
    Note: 2013→2018 vintage dropped (post-crash outlier, mean appreciation ~97%)
    Note: vintage_year included as a feature to control for market regime
    Note: 2015-2020 and 2018-2023 windows share 2 overlapping years — cluster SEs by tract

  Both validated on 2020 ACS with their respective appreciation windows.

Inputs required (all in the same folder):
    census_multivintage.csv     ACS 5-yr vintages 2013-2020  (from census_pull.py)
    business_density.csv        LODES job density by tract × year
    ZillowHousingData.csv       Zillow ZHVI wide format (ZIP level)
    zip_tract_crosswalk.csv     tract_id ↔ zip_code mapping
    training_final_v2.csv       original training data (city assignments + income filter)
    validation_final_v2.csv     original validation data (city assignments + income filter)

Usage:
    python build_all_datasets.py

Output files: training_1yr.csv, validation_1yr.csv, training_5yr.csv, validation_5yr.csv
"""

import pandas as pd
import numpy as np

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

BASE_FEATURES = [
    "median_income", "median_rent", "median_home_value",
    "share_college", "share_25_34", "homeownership_rate",
    "vacancy_rate", "share_pre1940", "share_white", "share_black",
    "share_hispanic", "poverty_rate", "rent_burden", "population",
    "arts_density", "food_density", "gentrify_density",
    "arts_job_share", "food_job_share",
    "city_sf", "city_oakland", "city_san_jose",
]

# ACS vintage years for each model
VINTAGES_1YR = list(range(2013, 2021))       # 2013-2020 (2020 = validation)
VINTAGES_5YR = [2013, 2015, 2018, 2020]      # 2020 = validation

# Zillow appreciation windows: {vintage_year: (begin_year, end_year)}
ZILLOW_1YR = {yr: (yr, yr + 1) for yr in VINTAGES_1YR}
ZILLOW_5YR = {2013: (2013, 2018), 2015: (2015, 2020),
              2018: (2018, 2023), 2020: (2020, 2025)}

# Nearest LODES year per vintage (update to exact mapping after lodes_multiyear_pull.py)
# Once all LODES years are pulled, replace every value with its key
LODES_YEAR_MAP = {
    2013: 2013, 2014: 2014, 2015: 2015, 2016: 2016,
    2017: 2017, 2018: 2018, 2019: 2019, 2020: 2020,
}
# Fallback if a LODES year is missing from business_density.csv:
LODES_FALLBACK = {2013: 2015, 2014: 2015, 2016: 2015, 2017: 2015,
                  2018: 2020, 2019: 2020}

# Known city median incomes (constant per city in the original data)
# Used to reconstruct below_city_median_income for intermediate vintage years
# Format: {city: {year: city_median_income}}
CITY_MEDIANS_KNOWN = {
    "Oakland":       {2010: 50311, 2015: 59438},
    "Other":         {2010: 87126, 2015: 93818},
    "San Francisco": {2010: 78774, 2015: 86536},
    "San Jose":      {2010: 81676, 2015: 88659},
}

# Winsorization percentiles
WINSOR_LO = 0.01
WINSOR_HI = 0.99

# ── LOAD INPUTS ───────────────────────────────────────────────────────────────

print("=" * 60)
print("Loading inputs...")
print("=" * 60)

acs    = pd.read_csv("census_multivintage.csv")
biz    = pd.read_csv("business_density.csv")
zillow = pd.read_csv("ZillowHousingData.csv")
xw     = pd.read_csv("zip_tract_crosswalk.csv")
orig_train = pd.read_csv("training_final_v2.csv")
orig_val   = pd.read_csv("validation_final_v2.csv")

for df in [acs, biz, orig_train, orig_val]:
    df["tract_id"] = df["tract_id"].astype(str).str.zfill(11)
xw["tract_id"]  = xw["tract_id"].astype(str).str.zfill(11)
xw["zip_code"]  = xw["zip_code"].astype(str).str.zfill(5)
zillow["zip_code"] = zillow["RegionName"].astype(str).str.zfill(5)

print(f"  ACS vintages loaded:   {sorted(acs['year'].unique().tolist())}")
print(f"  LODES years loaded:    {sorted(biz['year'].unique().tolist())}")
print(f"  Zillow ZIPs:           {len(zillow):,}")
print(f"  Crosswalk tracts:      {len(xw):,}")

# ── ZILLOW: BUILD APPRECIATION TABLE ─────────────────────────────────────────

print("\nBuilding Zillow appreciation table...")

# Collect all Jan values needed across both models
all_jan_years = sorted(set(
    [y for w in list(ZILLOW_1YR.values()) + list(ZILLOW_5YR.values()) for y in w]
))
print(f"  Jan years needed: {all_jan_years}")

jan_col_map = {}
for yr in all_jan_years:
    col = f"{yr}-01-31"
    if col in zillow.columns:
        jan_col_map[yr] = col
    else:
        print(f"  WARNING: {col} not in Zillow data — skipping")

# Build ZIP-level Jan value table
zip_jan = zillow[["zip_code"] + list(jan_col_map.values())].copy()
zip_jan = zip_jan.rename(columns={v: f"jan_{k}" for k, v in jan_col_map.items()})

# Merge to tract level via crosswalk
tract_jan = xw.merge(zip_jan, on="zip_code", how="left")
print(f"  Tracts with Jan 2015 coverage: {tract_jan['jan_2015'].notna().sum():,}")
print(f"  Tracts with Jan 2020 coverage: {tract_jan['jan_2020'].notna().sum():,}")

def compute_appreciation(df, begin_yr, end_yr):
    """Compute (jan_end - jan_begin) / jan_begin for a tract-level Jan table."""
    col_b = f"jan_{begin_yr}"
    col_e = f"jan_{end_yr}"
    if col_b not in df.columns or col_e not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return (df[col_e] - df[col_b]) / df[col_b]

# Compute all appreciation windows
for vintage, (b, e) in ZILLOW_1YR.items():
    tract_jan[f"appr_1yr_{vintage}"] = compute_appreciation(tract_jan, b, e)

for vintage, (b, e) in ZILLOW_5YR.items():
    tract_jan[f"appr_5yr_{vintage}"] = compute_appreciation(tract_jan, b, e)

print("\n  1-Year appreciation summary:")
for vintage in VINTAGES_1YR:
    col = f"appr_1yr_{vintage}"
    s   = tract_jan[col].dropna()
    print(f"    {vintage}→{vintage+1}: n={len(s):,}  mean={s.mean():.3f}  std={s.std():.3f}")

print("\n  5-Year appreciation summary:")
for vintage, (b, e) in ZILLOW_5YR.items():
    col = f"appr_5yr_{vintage}"
    s   = tract_jan[col].dropna()
    print(f"    {b}→{e}: n={len(s):,}  mean={s.mean():.3f}  std={s.std():.3f}")

# ── CITY ASSIGNMENTS AND INCOME FILTER ───────────────────────────────────────

print("\nBuilding city assignments and income filter...")

# Get city, city dummies, and below_city_median_income from original data
city_cols = ["tract_id", "city", "city_sf", "city_oakland", "city_san_jose"]

city_assign = (orig_train[orig_train["year"] == 2015][city_cols]
               .drop_duplicates("tract_id")
               .copy())

# Get 2020 city median incomes from validation file
city_medians_2020 = (orig_val.groupby("city")["city_median_income"]
                     .first().to_dict())
for city, med in city_medians_2020.items():
    CITY_MEDIANS_KNOWN.setdefault(city, {})[2020] = med
print(f"  City medians 2020: { {k: round(v) for k, v in city_medians_2020.items()} }")

def interpolate_city_median(city, year, known):
    """Linearly interpolate city median income for a given year."""
    yr_vals = known.get(city, {})
    if year in yr_vals:
        return yr_vals[year]
    known_years = sorted(yr_vals.keys())
    if not known_years:
        return np.nan
    if year < known_years[0]:
        return yr_vals[known_years[0]]
    if year > known_years[-1]:
        return yr_vals[known_years[-1]]
    # Linear interpolation between nearest brackets
    lo = max(y for y in known_years if y <= year)
    hi = min(y for y in known_years if y >= year)
    if lo == hi:
        return yr_vals[lo]
    t = (year - lo) / (hi - lo)
    return yr_vals[lo] + t * (yr_vals[hi] - yr_vals[lo])

# Build city-median lookup for all vintage years
city_median_by_year = {}
for vintage in set(list(VINTAGES_1YR) + list(VINTAGES_5YR)):
    city_median_by_year[vintage] = {
        city: interpolate_city_median(city, vintage, CITY_MEDIANS_KNOWN)
        for city in ["Oakland", "Other", "San Francisco", "San Jose"]
    }

# ── LODES HELPER ─────────────────────────────────────────────────────────────

biz_years_available = set(biz["year"].unique())
BIZ_COLS = ["tract_id", "arts_density", "food_density", "gentrify_density",
            "arts_job_share", "food_job_share"]

def get_biz(vintage_year):
    """Return business density for the appropriate LODES year."""
    lodes_yr = LODES_YEAR_MAP.get(vintage_year)
    if lodes_yr not in biz_years_available:
        lodes_yr = LODES_FALLBACK.get(vintage_year, 2015)
    sub = biz[biz["year"] == lodes_yr][[c for c in BIZ_COLS if c in biz.columns]].copy()
    sub["lodes_year"] = lodes_yr
    return sub

# ── CORE BUILD FUNCTION ───────────────────────────────────────────────────────

def build_vintage(vintage_year, appreciation_col, target_name):
    """
    Assemble one vintage-year slice: ACS + LODES + Zillow + city filter.
    Returns a DataFrame with BASE_FEATURES + target + metadata columns.
    """
    # ACS features for this vintage
    acs_v = acs[acs["year"] == vintage_year].copy()
    if len(acs_v) == 0:
        print(f"    WARNING: No ACS data for vintage {vintage_year}")
        return pd.DataFrame()

    # Zillow appreciation
    appr = tract_jan[["tract_id", appreciation_col, f"jan_{ZILLOW_1YR.get(vintage_year, ZILLOW_5YR.get(vintage_year, (vintage_year, vintage_year+1)))[0]}"]].copy()
    appr = appr.rename(columns={
        appreciation_col: target_name,
        f"jan_{ZILLOW_1YR.get(vintage_year, ZILLOW_5YR.get(vintage_year, (vintage_year, vintage_year+1)))[0]}": "jan_value_begin"
    })

    # LODES business density
    biz_v = get_biz(vintage_year)

    # City assignments
    merged = (acs_v
              .merge(appr,       on="tract_id", how="inner")
              .merge(biz_v,      on="tract_id", how="left")
              .merge(city_assign, on="tract_id", how="left"))

    # Reconstruct below_city_median_income using interpolated city medians
    city_med = city_median_by_year[vintage_year]
    merged["city_median_income_est"] = merged["city"].map(city_med)
    merged["below_city_median_income"] = (
        merged["median_income"] < merged["city_median_income_est"]
    ).astype(int)

    # Filter: below city median income + valid target
    before = len(merged)
    merged = merged[merged["below_city_median_income"] == 1].dropna(subset=[target_name]).copy()
    merged["vintage_year"] = vintage_year
    merged["lodes_year"]   = biz_v["lodes_year"].iloc[0] if len(biz_v) > 0 else np.nan

    return merged

# ── BUILD 1-YEAR MODEL DATASETS ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("Building 1-Year Appreciation datasets...")
print("=" * 60)

TARGET_1YR = "appreciation_1yr"
parts_1yr_train = []

for vintage in range(2013, 2019):   # 2013-2018 training (2019→2020 dropped: COVID onset)
    appr_col = f"appr_1yr_{vintage}"
    df = build_vintage(vintage, appr_col, TARGET_1YR)
    if len(df) > 0:
        print(f"  {vintage}→{vintage+1}: {len(df):,} rows  "
              f"mean={df[TARGET_1YR].mean():.3f}  "
              f"std={df[TARGET_1YR].std():.3f}  "
              f"(LODES {df['lodes_year'].iloc[0]:.0f})")
        parts_1yr_train.append(df)

train_1yr = pd.concat(parts_1yr_train, ignore_index=True)

# Winsorize training target
lo = train_1yr[TARGET_1YR].quantile(WINSOR_LO)
hi = train_1yr[TARGET_1YR].quantile(WINSOR_HI)
n_clip = ((train_1yr[TARGET_1YR] < lo) | (train_1yr[TARGET_1YR] > hi)).sum()
train_1yr[TARGET_1YR] = train_1yr[TARGET_1YR].clip(lower=lo, upper=hi)
print(f"\n  Winsorized {n_clip} rows [{lo:.3f}, {hi:.3f}]")
print(f"  Total training rows: {len(train_1yr):,}  |  "
      f"Unique tracts: {train_1yr['tract_id'].nunique():,}")

# Validation: 2020 vintage → 2020→2021
val_1yr = build_vintage(2020, "appr_1yr_2020", TARGET_1YR)
print(f"\n  Validation (2020→2021): {len(val_1yr):,} rows  "
      f"mean={val_1yr[TARGET_1YR].mean():.3f}  "
      f"std={val_1yr[TARGET_1YR].std():.3f}")

# ── BUILD 5-YEAR MODEL DATASETS ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("Building 5-Year Appreciation datasets...")
print("=" * 60)

TARGET_5YR = "appreciation_5yr"
parts_5yr_train = []

for vintage, (b, e) in [(2015,(2015,2020)), (2018,(2018,2023))]:  # 2013→2018 dropped: post-crash outlier
    appr_col = f"appr_5yr_{vintage}"
    df = build_vintage(vintage, appr_col, TARGET_5YR)
    if len(df) > 0:
        print(f"  {vintage} ACS → {b}→{e}: {len(df):,} rows  "
              f"mean={df[TARGET_5YR].mean():.3f}  "
              f"std={df[TARGET_5YR].std():.3f}  "
              f"(LODES {df['lodes_year'].iloc[0]:.0f})")
        parts_5yr_train.append(df)

train_5yr = pd.concat(parts_5yr_train, ignore_index=True)

# Winsorize training target
lo = train_5yr[TARGET_5YR].quantile(WINSOR_LO)
hi = train_5yr[TARGET_5YR].quantile(WINSOR_HI)
n_clip = ((train_5yr[TARGET_5YR] < lo) | (train_5yr[TARGET_5YR] > hi)).sum()
train_5yr[TARGET_5YR] = train_5yr[TARGET_5YR].clip(lower=lo, upper=hi)
print(f"\n  Winsorized {n_clip} rows [{lo:.3f}, {hi:.3f}]")
print(f"  Total training rows: {len(train_5yr):,}  |  "
      f"Unique tracts: {train_5yr['tract_id'].nunique():,}")
print(f"\n  NOTE: outcome windows 2015-2020 and 2018-2023 share 2 overlapping years.")
print(f"  vintage_year is included as a feature to control for market regime.")
print(f"  Cluster standard errors by tract_id in write-up.")

# Validation: 2020 vintage → 2020→2025
val_5yr = build_vintage(2020, "appr_5yr_2020", TARGET_5YR)
print(f"\n  Validation (2020→2025): {len(val_5yr):,} rows  "
      f"mean={val_5yr[TARGET_5YR].mean():.3f}  "
      f"std={val_5yr[TARGET_5YR].std():.3f}")

# ── SAVE ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Saving datasets...")
print("=" * 60)

META_COLS = ["tract_id", "vintage_year", "lodes_year", "jan_value_begin",
             "city", "city_median_income_est", "below_city_median_income"]

def save_dataset(df, target_col, filename, extra_features=None):
    features = BASE_FEATURES + (extra_features or [])
    out_cols = META_COLS + [target_col] + [f for f in features if f in df.columns]
    out_cols = list(dict.fromkeys(out_cols))   # deduplicate while preserving order
    missing  = [f for f in features if f not in df.columns]
    if missing:
        print(f"  WARNING: {filename} missing features: {missing}")
    df[[c for c in out_cols if c in df.columns]].to_csv(filename, index=False)
    print(f"  Saved {filename:<35} {len(df):,} rows  target={target_col}")

# 1-year: no extra features
save_dataset(train_1yr, TARGET_1YR, "training_1yr.csv")
save_dataset(val_1yr,   TARGET_1YR, "validation_1yr.csv")

# 5-year: vintage_year added as a feature to control for market regime
save_dataset(train_5yr, TARGET_5YR, "training_5yr.csv",   extra_features=["vintage_year"])
save_dataset(val_5yr,   TARGET_5YR, "validation_5yr.csv", extra_features=["vintage_year"])

# ── SUMMARY ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"\n  1-Year Model")
print(f"    Training:   {len(train_1yr):,} rows  "
      f"({train_1yr['tract_id'].nunique()} tracts × "
      f"{train_1yr['vintage_year'].nunique()} vintages)")
print(f"    Validation: {len(val_1yr):,} rows  (2020→2021)")
print(f"    Target range: [{train_1yr[TARGET_1YR].min():.3f}, "
      f"{train_1yr[TARGET_1YR].max():.3f}]")
print(f"    Target mean:  {train_1yr[TARGET_1YR].mean():.3f}  "
      f"std={train_1yr[TARGET_1YR].std():.3f}")

print(f"\n  5-Year Model")
print(f"    Training:   {len(train_5yr):,} rows  "
      f"({train_5yr['tract_id'].nunique()} tracts × "
      f"{train_5yr['vintage_year'].nunique()} vintages)")
print(f"    Validation: {len(val_5yr):,} rows  (2020→2025)")
print(f"    Target range: [{train_5yr[TARGET_5YR].min():.3f}, "
      f"{train_5yr[TARGET_5YR].max():.3f}]")
print(f"    Target mean:  {train_5yr[TARGET_5YR].mean():.3f}  "
      f"std={train_5yr[TARGET_5YR].std():.3f}")

print("\nDone. Run the model cells using:")
print("  1-year: training_1yr.csv / validation_1yr.csv  target=appreciation_1yr")
print("  5-year: training_5yr.csv / validation_5yr.csv  target=appreciation_5yr")
