"""
build_multivintage_dataset.py
------------------------------
Merges multi-vintage ACS features, 1-year Zillow appreciation targets, and
business density data into training and validation datasets.

Inputs (must be in the same folder):
    census_multivintage.csv     — ACS 5-year vintages 2013-2020 (from census_pull.py)
    zillow_annual.csv           — 1-year appreciation per tract per year (from zillow_annual.py)
    business_density.csv        — arts/food job density by tract and year
    training_final_v2.csv       — original training data (for city dummies, income filter)
    validation_final_v2.csv     — original validation data (for city dummies, income filter)

Outputs:
    training_multivintage.csv   — 2013-2019 vintages, labeled with 1-year appreciation
    validation_multivintage.csv — 2020 vintage, labeled with 2020→2021 appreciation

Business density note:
    LODES WAC data is only available for 2010, 2015, 2020 in this pipeline.
    For intermediate years, the nearest available LODES year is used:
        2013 → 2015 LODES
        2014 → 2015 LODES
        2015 → 2015 LODES
        2016 → 2015 LODES
        2017 → 2015 LODES (or 2020 if you prefer — change LODES_YEAR_MAP below)
        2018 → 2020 LODES
        2019 → 2020 LODES
        2020 → 2020 LODES
    To improve this, pull LODES WAC data for 2013-2019 from:
        https://lehd.ces.census.gov/data/lodes/LODES8/ca/wac/
    and re-run business_density.py for each year, then update business_density.csv.

Usage:
    python build_multivintage_dataset.py
"""

import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Nearest available LODES year for each ACS vintage
LODES_YEAR_MAP = {
    2013: 2015,
    2014: 2015,
    2015: 2015,
    2016: 2015,
    2017: 2015,
    2018: 2020,
    2019: 2020,
    2020: 2020,
}

BASE_FEATURES = [
    "median_income", "median_rent", "median_home_value",
    "share_college", "share_25_34", "homeownership_rate",
    "vacancy_rate", "share_pre1940", "share_white", "share_black",
    "share_hispanic", "poverty_rate", "rent_burden", "population",
    "arts_density", "food_density", "gentrify_density",
    "arts_job_share", "food_job_share",
    "city_sf", "city_oakland", "city_san_jose",
]

TARGET = "appreciation_1yr"

# ── LOAD ──────────────────────────────────────────────────────────────────────

print("Loading inputs...")
acs      = pd.read_csv("census_multivintage.csv")
zillow   = pd.read_csv("zillow_annual.csv")
biz      = pd.read_csv("business_density.csv")
train_v2 = pd.read_csv("training_final_v2.csv")
val_v2   = pd.read_csv("validation_final_v2.csv")

# Standardize tract IDs
for df in [acs, zillow, biz, train_v2, val_v2]:
    df["tract_id"] = df["tract_id"].astype(str).str.zfill(11)

print(f"  ACS vintages: {acs['year'].unique().tolist()}")
print(f"  Zillow years: {sorted(zillow['year'].unique().tolist())}")
print(f"  LODES years:  {sorted(biz['year'].unique().tolist())}")

# ── CITY DUMMIES AND INCOME FILTER ────────────────────────────────────────────
# Pull city dummies and below_city_median_income from original training/val files
# Use 2015 for city dummy assignment (stable — same tracts, same cities)

city_info = (train_v2[train_v2["year"] == 2015]
             [["tract_id", "city_sf", "city_oakland", "city_san_jose",
               "below_city_median_income"]]
             .drop_duplicates("tract_id"))

# For validation, use 2020 city info
city_info_val = (val_v2[["tract_id", "city_sf", "city_oakland", "city_san_jose",
                          "below_city_median_income"]]
                 .drop_duplicates("tract_id"))

# ── BUSINESS DENSITY MERGE ────────────────────────────────────────────────────

def get_biz_for_vintage(vintage_year, biz_df, year_map):
    """Return business density rows for the nearest available LODES year."""
    lodes_year = year_map.get(vintage_year, 2015)
    sub = biz_df[biz_df["year"] == lodes_year].copy()
    sub = sub.rename(columns={"year": "lodes_year"})
    return sub

# ── BUILD TRAINING SET (2013-2019 vintages) ───────────────────────────────────

print("\nBuilding training set (2013-2019 vintages)...")
train_parts = []

for vintage in range(2013, 2020):
    acs_v = acs[acs["year"] == vintage].copy()
    if len(acs_v) == 0:
        print(f"  WARNING: No ACS data for vintage {vintage} — skipping")
        continue

    # Zillow: appreciation for Jan vintage → Jan (vintage+1)
    zillow_v = zillow[zillow["year"] == vintage][["tract_id", "jan_value", "appreciation_1yr"]]

    # Business density: nearest LODES year
    biz_v = get_biz_for_vintage(vintage, biz, LODES_YEAR_MAP)
    biz_cols = ["tract_id", "arts_density", "food_density", "gentrify_density",
                "arts_job_share", "food_job_share"]
    biz_v = biz_v[[c for c in biz_cols if c in biz_v.columns]]

    # Merge
    merged = (acs_v
              .merge(zillow_v, on="tract_id", how="inner")
              .merge(biz_v,    on="tract_id", how="left")
              .merge(city_info, on="tract_id", how="left"))

    n_before = len(merged)
    merged = merged[merged["below_city_median_income"] == 1].dropna(subset=[TARGET]).copy()
    merged["lodes_year"] = LODES_YEAR_MAP[vintage]

    print(f"  {vintage}: {n_before} tracts → {len(merged)} below-median eligible "
          f"(LODES year: {LODES_YEAR_MAP[vintage]})")
    train_parts.append(merged)

train_out = pd.concat(train_parts, ignore_index=True)

# ── WINSORIZE TARGET ──────────────────────────────────────────────────────────
# Winsorize at 1st and 99th percentile (1-year appreciation can have outliers)
lo = train_out[TARGET].quantile(0.01)
hi = train_out[TARGET].quantile(0.99)
n_clipped = ((train_out[TARGET] < lo) | (train_out[TARGET] > hi)).sum()
train_out[TARGET] = train_out[TARGET].clip(lower=lo, upper=hi)
print(f"\nWinsorized {n_clipped} extreme appreciation values [{lo:.3f}, {hi:.3f}]")

# ── TRAINING SUMMARY ──────────────────────────────────────────────────────────
print(f"\n=== Training Set Summary ===")
print(f"Total rows: {len(train_out):,}")
print(f"Unique tracts: {train_out['tract_id'].nunique():,}")
print(f"Vintage year distribution:")
print(train_out.groupby("year")[TARGET].agg(["count","mean","std"]).round(4).to_string())

# ── BUILD VALIDATION SET (2020 vintage → 2020→2021 appreciation) ──────────────
print("\nBuilding validation set (2020 vintage, 2020→2021 appreciation)...")

acs_2020    = acs[acs["year"] == 2020].copy()
zillow_2020 = zillow[zillow["year"] == 2020][["tract_id", "jan_value", "appreciation_1yr"]]
biz_2020    = get_biz_for_vintage(2020, biz, LODES_YEAR_MAP)
biz_cols    = ["tract_id", "arts_density", "food_density", "gentrify_density",
               "arts_job_share", "food_job_share"]
biz_2020    = biz_2020[[c for c in biz_cols if c in biz_2020.columns]]

val_out = (acs_2020
           .merge(zillow_2020,   on="tract_id", how="inner")
           .merge(biz_2020,      on="tract_id", how="left")
           .merge(city_info_val, on="tract_id", how="left"))

val_out = val_out[val_out["below_city_median_income"] == 1].dropna(subset=[TARGET]).copy()
val_out["lodes_year"] = 2020

print(f"Validation rows: {len(val_out):,}")
print(f"Validation appreciation (2020→2021): mean={val_out[TARGET].mean():.4f}  "
      f"std={val_out[TARGET].std():.4f}")

# ── FEATURE AVAILABILITY CHECK ────────────────────────────────────────────────

missing_train = [f for f in BASE_FEATURES if f not in train_out.columns]
missing_val   = [f for f in BASE_FEATURES if f not in val_out.columns]
if missing_train:
    print(f"\nWARNING: Features missing from training: {missing_train}")
if missing_val:
    print(f"WARNING: Features missing from validation: {missing_val}")

avail = [f for f in BASE_FEATURES if f in train_out.columns and f in val_out.columns]
print(f"\nFeatures available in both sets: {len(avail)} / {len(BASE_FEATURES)}")

# ── SAVE ──────────────────────────────────────────────────────────────────────

out_cols = ["tract_id", "year", TARGET, "jan_value", "lodes_year"] + avail

train_out[out_cols].to_csv("training_multivintage.csv", index=False)
val_out[[c for c in out_cols if c in val_out.columns]].to_csv(
    "validation_multivintage.csv", index=False)

print(f"\nSaved training_multivintage.csv   ({len(train_out):,} rows)")
print(f"Saved validation_multivintage.csv ({len(val_out):,} rows)")
print("\nNext step: run the model cells using these files.")
print("  Target column:  appreciation_1yr")
print("  Features:       BASE_FEATURES (no permits — compatible with all tracts)")
print("  Validation:     2020→2021 appreciation (Jan 2021 Zillow value available)")
