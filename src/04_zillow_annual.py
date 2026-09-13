"""
zillow_annual.py
----------------
Extracts 1-year appreciation rates per census tract per year from the
Zillow ZHVI ZIP-code data, using a ZIP→tract crosswalk.

Inputs (must be in the same folder):
    ZillowHousingData.csv       — Zillow ZHVI wide format, one row per ZIP
    zip_tract_crosswalk.csv     — columns: tract_id, zip_code

Output:
    zillow_annual.csv           — long format: tract_id, year, jan_value, appreciation_1yr

Appreciation is defined as:
    (jan_value[year+1] - jan_value[year]) / jan_value[year]

For example, the row with year=2015 gives appreciation from Jan 2015 → Jan 2016.
This means the row's features (2015 ACS vintage) predict the following year's price change.

Years produced: 2012 through 2024 (appreciation_1yr defined for 2012–2024,
using Jan values from 2012 through 2025).

Usage:
    python zillow_annual.py
"""

import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────

ZILLOW_FILE    = "ZillowHousingData.csv"
CROSSWALK_FILE = "zip_tract_crosswalk.csv"
OUTPUT_FILE    = "zillow_annual.csv"

# Range of Jan values to extract (need year+1 to compute appreciation for each year)
JAN_YEARS = list(range(2012, 2026))   # 2012–2025 inclusive

# ── LOAD ──────────────────────────────────────────────────────────────────────

print("Loading Zillow data...")
zillow = pd.read_csv(ZILLOW_FILE)
zillow["zip_code"] = zillow["RegionName"].astype(str).str.zfill(5)

print("Loading crosswalk...")
xw = pd.read_csv(CROSSWALK_FILE)
xw["tract_id"] = xw["tract_id"].astype(str).str.zfill(11)
xw["zip_code"] = xw["zip_code"].astype(str).str.zfill(5)

# ── EXTRACT JANUARY VALUES ────────────────────────────────────────────────────

jan_cols = {}
for yr in JAN_YEARS:
    col = f"{yr}-01-31"
    if col in zillow.columns:
        jan_cols[yr] = col
    else:
        print(f"  WARNING: {col} not found in Zillow data — skipping year {yr}")

print(f"January columns found: {list(jan_cols.keys())}")

# Build ZIP-level annual table
zip_annual = zillow[["zip_code"] + list(jan_cols.values())].copy()
zip_annual = zip_annual.rename(columns={v: f"jan_{k}" for k, v in jan_cols.items()})

# ── MERGE CROSSWALK → ZIP DATA ────────────────────────────────────────────────

print("Merging crosswalk with Zillow...")
merged = xw.merge(zip_annual, on="zip_code", how="left")
print(f"  Tracts in crosswalk:         {len(xw):,}")
print(f"  Tracts with any Zillow data: {merged['jan_2015'].notna().sum():,}")

# ── COMPUTE 1-YEAR APPRECIATION ───────────────────────────────────────────────

print("Computing 1-year appreciation rates...")
records = []

for yr in JAN_YEARS[:-1]:   # 2012–2024 (need yr+1 to compute appreciation)
    next_yr = yr + 1
    if yr not in jan_cols or next_yr not in jan_cols:
        continue

    col_begin = f"jan_{yr}"
    col_end   = f"jan_{next_yr}"

    if col_begin not in merged.columns or col_end not in merged.columns:
        continue

    sub = merged[["tract_id", col_begin, col_end]].copy()
    sub = sub.rename(columns={col_begin: "jan_value", col_end: "jan_value_next"})
    sub["year"] = yr
    sub["appreciation_1yr"] = (sub["jan_value_next"] - sub["jan_value"]) / sub["jan_value"]

    # Drop rows where either value is missing
    valid = sub.dropna(subset=["jan_value", "jan_value_next"])
    records.append(valid[["tract_id", "year", "jan_value", "appreciation_1yr"]])

result = pd.concat(records, ignore_index=True)

# ── SUMMARY ───────────────────────────────────────────────────────────────────

print("\n=== 1-Year Appreciation by Year ===")
summary = result.groupby("year")["appreciation_1yr"].agg(["count", "mean", "std"])
summary.columns = ["n_tracts", "mean_appr", "std_appr"]
print(summary.round(4).to_string())

print(f"\nTotal rows: {len(result):,}")
print(f"Unique tracts: {result['tract_id'].nunique():,}")

# ── SAVE ──────────────────────────────────────────────────────────────────────

result.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {OUTPUT_FILE}")
print("Columns: tract_id, year, jan_value, appreciation_1yr")
print("\nKey: year=2015 → appreciation from Jan 2015 → Jan 2016")
print("     Use with 2015 ACS vintage features to predict next-year price change.")
