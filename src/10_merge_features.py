"""
merge_features.py
-----------------
Merges Census ACS, LODES business density, and Zillow ZHVI data into
a single feature matrix for the gentrification prediction model.

Outputs two files:
  feature_matrix_wide.csv — one row per tract, all years side by side
  feature_matrix_long.csv — one row per (tract × time_window) with
                            train/validate/test labels and target variable

Time windows:
  TRAIN    — 2010 features → 2010-2015 Zillow appreciation
  VALIDATE — 2015 features → 2015-2019 Zillow appreciation
  TEST     — 2022/2024 features → 2022-present Zillow appreciation

Usage:
    python merge_features.py

Requirements:
    pip install pandas
"""

import pandas as pd
import numpy as np
import os

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

CENSUS_WIDE       = "census_wide.csv"
LODES_WIDE        = "business_density_wide.csv"
ZILLOW_CSV        = "zillow_zhvi.csv"       # Set to None if not yet downloaded
OUTPUT_WIDE       = "feature_matrix_wide.csv"
OUTPUT_LONG       = "feature_matrix_long.csv"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def normalize_tract_id(series):
    """Ensures tract_id is a zero-padded 11-digit string."""
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11)


# ── LOAD DATASETS ─────────────────────────────────────────────────────────────

print("Loading Census ACS data...")
census = pd.read_csv(CENSUS_WIDE)
census["tract_id"] = normalize_tract_id(census["tract_id"])
print(f"  {len(census):,} tracts, {census.shape[1]} columns")

print("Loading LODES business density data...")
lodes = pd.read_csv(LODES_WIDE)
lodes["tract_id"] = normalize_tract_id(lodes["tract_id"])
print(f"  {len(lodes):,} tracts, {lodes.shape[1]} columns")


# ── MERGE CENSUS + LODES ──────────────────────────────────────────────────────

print("\nMerging Census + LODES on tract_id...")
df = census.merge(lodes, on="tract_id", how="left")
print(f"  Merged shape: {df.shape}")
print(f"  Tracts with LODES data: {lodes['tract_id'].isin(df['tract_id']).sum():,}")
print(f"  Tracts missing LODES (census-only): {df['arts_density_2010'].isna().sum():,}")


# ── LOAD AND MERGE ZILLOW ─────────────────────────────────────────────────────
#
# Zillow ZHVI is available at the ZIP code level. To merge it here you need
# a ZIP-to-tract crosswalk (zip_tract_crosswalk.csv) already built in Colab.
#
# Expected Zillow CSV format after preprocessing:
#   tract_id | zhvi_2010 | zhvi_2015 | zhvi_2019 | zhvi_2022 | zhvi_2025
#
# Appreciation is computed as: (zhvi_end - zhvi_start) / zhvi_start

if ZILLOW_CSV and os.path.exists(ZILLOW_CSV):
    print("\nLoading Zillow ZHVI data...")
    zillow = pd.read_csv(ZILLOW_CSV)
    zillow["tract_id"] = normalize_tract_id(zillow["tract_id"])

    # Compute appreciation for each time window
    if "zhvi_2010" in zillow.columns and "zhvi_2015" in zillow.columns:
        zillow["appreciation_2010_2015"] = (zillow["zhvi_2015"] - zillow["zhvi_2010"]) / zillow["zhvi_2010"]
    if "zhvi_2015" in zillow.columns and "zhvi_2019" in zillow.columns:
        zillow["appreciation_2015_2019"] = (zillow["zhvi_2019"] - zillow["zhvi_2015"]) / zillow["zhvi_2015"]
    if "zhvi_2022" in zillow.columns and "zhvi_2025" in zillow.columns:
        zillow["appreciation_2022_2025"] = (zillow["zhvi_2025"] - zillow["zhvi_2022"]) / zillow["zhvi_2022"]

    df = df.merge(zillow, on="tract_id", how="left")
    print(f"  Merged shape after Zillow: {df.shape}")
else:
    print("\nZillow data not found — skipping. Add zillow_zhvi.csv to include target variable.")
    print("  (feature_matrix_wide.csv will still be saved without target columns)")


# ── SAVE WIDE FORMAT ──────────────────────────────────────────────────────────

df.to_csv(OUTPUT_WIDE, index=False)
print(f"\nSaved {OUTPUT_WIDE} ({len(df):,} tracts × {df.shape[1]} columns)")


# ── BUILD LONG FORMAT (one row per tract × time window) ───────────────────────
#
# Each row represents one observation period with:
#   - Features from the START of the window
#   - Target = appreciation over the window (if Zillow available)
#   - split = train / validate / test

print("\nBuilding long-format feature matrix...")

windows = [
    {
        "split":       "train",
        "feat_year":   "2010",
        "lodes_year":  "2010",
        "target_col":  "appreciation_2010_2015",
    },
    {
        "split":       "validate",
        "feat_year":   "2015",
        "lodes_year":  "2015",
        "target_col":  "appreciation_2015_2019",
    },
    {
        "split":       "test",
        "feat_year":   "2020",       # Use 2020 census (2024 not yet re-pulled)
        "lodes_year":  "2022",
        "target_col":  "appreciation_2022_2025",
    },
]

# Census features to include per window
CENSUS_FEATURES = [
    "population", "median_income", "median_rent", "median_home_value",
    "share_college", "share_25_34", "homeownership_rate", "vacancy_rate",
    "share_pre1940", "share_white", "share_black", "share_hispanic",
    "poverty_rate", "rent_burden",
]

# LODES features to include per window
LODES_FEATURES = [
    "arts_density", "food_density", "gentrify_density",
    "arts_job_share", "food_job_share",
]

long_rows = []

for w in windows:
    fy  = w["feat_year"]
    ly  = w["lodes_year"]
    tgt = w["target_col"]

    row_df = df[["tract_id"]].copy()
    row_df["split"]     = w["split"]
    row_df["feat_year"] = int(fy)

    # Pull census features for this window's year
    for feat in CENSUS_FEATURES:
        col = f"{feat}_{fy}"
        row_df[feat] = df[col] if col in df.columns else np.nan

    # Pull LODES features for this window's year
    for feat in LODES_FEATURES:
        col = f"{feat}_{ly}"
        row_df[feat] = df[col] if col in df.columns else np.nan

    # Pull target variable if Zillow data is present
    row_df["target_appreciation"] = df[tgt] if tgt in df.columns else np.nan

    long_rows.append(row_df)

df_long = pd.concat(long_rows, ignore_index=True)
df_long.to_csv(OUTPUT_LONG, index=False)
print(f"Saved {OUTPUT_LONG} ({len(df_long):,} rows × {df_long.shape[1]} columns)")


# ── SUMMARY ───────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("FEATURE MATRIX SUMMARY")
print("="*55)
for split in ["train", "validate", "test"]:
    sub = df_long[df_long["split"] == split]
    n_tracts    = len(sub)
    n_complete  = sub[CENSUS_FEATURES + LODES_FEATURES].dropna().shape[0]
    has_target  = sub["target_appreciation"].notna().sum()
    print(f"\n  {split.upper()}")
    print(f"    Tracts total:       {n_tracts:,}")
    print(f"    Fully complete:     {n_complete:,}")
    print(f"    With target var:    {has_target:,}")

print(f"\n  Total features per row: {len(CENSUS_FEATURES) + len(LODES_FEATURES)}")
print(f"    Census features:  {len(CENSUS_FEATURES)}")
print(f"    LODES features:   {len(LODES_FEATURES)}")

print("\nNext steps:")
print("  1. Download Zillow ZHVI and preprocess to zillow_zhvi.csv")
print("  2. Re-run census_pull.py with YEARS=[2010,2015,2020,2024]")
print("     then update TEST window feat_year to '2024'")
print("  3. Run this script again to include the target variable")
print("  4. Add walkscore_features.csv and HMDA features when ready")
