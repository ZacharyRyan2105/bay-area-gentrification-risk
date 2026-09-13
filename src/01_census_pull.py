"""
census_pull.py
--------------
Pulls all ACS 5-year estimate variables needed for the gentrification
prediction project for Bay Area census tracts.

Set MODE below to control what gets pulled:

  MODE = "all"      — pulls 2010, 2015, 2020, 2024 (full training/validation/test set)
  MODE = "test"     — pulls 2024 only (test set features, saves to separate files)
  MODE = "test_avg" — pulls 2021, 2022, 2023, 2024 and averages across years
                      (saves to census_2021-2024_wide.csv, does not touch census_wide.csv)

Usage:
    python census_pull.py
    OR paste into Google Colab cell by cell

Requirements:
    pip install requests pandas

Before running:
    Get a free Census API key at: api.census.gov/data/key_signup.html
    Paste it into CENSUS_API_KEY below.

Output (MODE = "all"):
    census_features.csv          — one row per tract per year
    census_wide.csv              — one row per tract with year-suffixed columns

Output (MODE = "test"):
    census_test_features.csv     — one row per tract for 2024
    census_test_wide.csv         — one row per tract with _2024 suffixed columns

Output (MODE = "test_avg"):
    census_2021-2024_wide.csv    — one row per tract, each column averaged across 2021-2024
"""

import os
import requests
import pandas as pd
import time

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "")

# ── MODE ──────────────────────────────────────────────────────────────────────
# "all"      → pulls 2010, 2015, 2020, 2024  (overwrites census_wide.csv)
# "test"     → pulls 2024 only               (saves to census_test_wide.csv)
# "test_avg" → pulls 2021-2024, averages     (saves to census_2021-2024_wide.csv)

MODE = "multivintage"

# Bay Area county FIPS codes (without state prefix)
BAY_AREA_COUNTIES = ["001", "013", "075", "081", "085"]
#                    Alameda  Contra  SF     San    Santa
#                             Costa          Mateo  Clara

STATE = "06"  # California

# Years controlled by MODE
if MODE == "test":
    YEARS = [2024]
elif MODE == "test_avg":
    YEARS = [2021, 2022, 2023, 2024]
elif MODE == "multivintage":
    # 7 annual 5-year ACS vintages for multi-vintage training
    # Each vintage predicts 1-year Zillow appreciation for that year → next year
    # 2020 included for validation set construction
    YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
else:
    YEARS = [2010, 2015, 2020, 2024]

# ── ACS VARIABLES ─────────────────────────────────────────────────────────────
# Each entry: API variable code -> readable column name
# All pulled at census tract level for Bay Area counties

ACS_VARS = {
    # Population
    "B01003_001E": "population",

    # Age 25-34 (young professional cohort)
    # Male: 25-29 + 30-34
    "B01001_010E": "male_25_29",
    "B01001_011E": "male_30_34",
    # Female: 25-29 + 30-34
    "B01001_034E": "female_25_29",
    "B01001_035E": "female_30_34",

    # Education (population 25+) — B15002 works across 2010, 2015, 2020
    "B15002_001E": "edu_total",          # Total pop 25+
    "B15002_015E": "edu_male_bach",      # Male - Bachelor's
    "B15002_016E": "edu_male_masters",   # Male - Master's
    "B15002_017E": "edu_male_prof",      # Male - Professional
    "B15002_018E": "edu_male_doc",       # Male - Doctorate
    "B15002_032E": "edu_female_bach",    # Female - Bachelor's
    "B15002_033E": "edu_female_masters", # Female - Master's
    "B15002_034E": "edu_female_prof",    # Female - Professional
    "B15002_035E": "edu_female_doc",     # Female - Doctorate

    # Income & poverty
    "B19013_001E": "median_income",
    "B17001_001E": "poverty_total",   # Total with poverty status determined
    "B17001_002E": "poverty_below",   # Below poverty line

    # Housing costs
    "B25064_001E": "median_rent",
    "B25077_001E": "median_home_value",

    # Homeownership & vacancy
    "B25003_001E": "tenure_total",    # Total occupied units
    "B25003_002E": "tenure_owner",    # Owner-occupied
    "B25002_001E": "units_total",     # Total housing units
    "B25002_003E": "units_vacant",    # Vacant units

    # Housing vintage (age of stock)
    "B25034_001E": "vintage_total",   # Total units
    "B25034_010E": "vintage_pre1940", # Built 1939 or earlier

    # Race & ethnicity
    "B02001_001E": "race_total",
    "B02001_002E": "race_white",      # White alone
    "B02001_003E": "race_black",      # Black alone
    "B03003_001E": "hispanic_total",
    "B03003_003E": "hispanic_yes",    # Hispanic or Latino
}

# ── FETCH FUNCTION ────────────────────────────────────────────────────────────

def fetch_acs(year, county, variables):
    """
    Fetches ACS 5-year estimates for a single county and year.
    Returns a DataFrame or None on error.
    """
    var_string = ",".join(["NAME"] + variables)
    key_param = f"&key={CENSUS_API_KEY}" if CENSUS_API_KEY else ""
    url = (
        f"https://api.census.gov/data/{year}/acs/acs5"
        f"?get={var_string}"
        f"&for=tract:*"
        f"&in=state:{STATE}%20county:{county}"
        f"{key_param}"
    )

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data[1:], columns=data[0])
            return df
        else:
            print(f"  Error {resp.status_code} for county {county}, year {year}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  Request failed for county {county}, year {year}: {e}")
        print(f"  Raw response: {resp.text[:200]}")
        return None


# ── MAIN PULL LOOP ────────────────────────────────────────────────────────────

variables = list(ACS_VARS.keys())
all_dfs = []

for year in YEARS:
    print(f"\nPulling {year} ACS data...")
    year_dfs = []

    for county in BAY_AREA_COUNTIES:
        print(f"  County {county}...", end=" ")
        df = fetch_acs(year, county, variables)
        if df is not None:
            year_dfs.append(df)
            print(f"  {len(df)} tracts")
        else:
            print("  FAILED")
        time.sleep(0.5)  # Be polite to the Census API

    if year_dfs:
        year_df = pd.concat(year_dfs, ignore_index=True)
        year_df["year"] = year
        all_dfs.append(year_df)
        print(f"  Total {year} tracts: {len(year_df)}")

# ── COMBINE AND CLEAN ─────────────────────────────────────────────────────────

if not all_dfs:
    print("\nNo data was collected — all requests failed.")
    print("The 2024 ACS requires an API key. Sign up at:")
    print("  https://api.census.gov/data/key_signup.html")
    print("Then paste your key into CENSUS_API_KEY at the top of this script.")
    exit()

print("\nCombining all years...")
df = pd.concat(all_dfs, ignore_index=True)

# Build 11-digit tract GEOID
df["tract_id"] = (
    df["state"].astype(str).str.zfill(2) +
    df["county"].astype(str).str.zfill(3) +
    df["tract"].astype(str).str.zfill(6)
)

# Rename API variable columns to readable names
df = df.rename(columns=ACS_VARS)

# Drop raw geographic component columns (keep tract_id instead)
df = df.drop(columns=["NAME", "state", "county", "tract"], errors="ignore")

# Convert all numeric columns (Census API returns everything as strings)
numeric_cols = list(ACS_VARS.values())
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace Census null sentinel values (-666666666) with NaN
df[numeric_cols] = df[numeric_cols].replace(-666666666, pd.NA)
df[numeric_cols] = df[numeric_cols].replace(-999999999, pd.NA)

# ── ENGINEER DERIVED FEATURES ─────────────────────────────────────────────────

print("Engineering derived features...")

# Share with bachelor's degree or higher (combining male + female from B15002)
df["share_college"] = (
    (df["edu_male_bach"] + df["edu_male_masters"] + df["edu_male_prof"] + df["edu_male_doc"] +
     df["edu_female_bach"] + df["edu_female_masters"] + df["edu_female_prof"] + df["edu_female_doc"])
    / df["edu_total"]
)

# Share aged 25-34
df["share_25_34"] = (
    (df["male_25_29"] + df["male_30_34"] + df["female_25_29"] + df["female_30_34"])
    / df["population"]
)

# Homeownership rate
df["homeownership_rate"] = df["tenure_owner"] / df["tenure_total"]

# Vacancy rate
df["vacancy_rate"] = df["units_vacant"] / df["units_total"]

# Share of housing built before 1940
df["share_pre1940"] = df["vintage_pre1940"] / df["vintage_total"]

# Share white non-Hispanic
df["share_white"] = df["race_white"] / df["race_total"]

# Share Black
df["share_black"] = df["race_black"] / df["race_total"]

# Share Hispanic
df["share_hispanic"] = df["hispanic_yes"] / df["hispanic_total"]

# Poverty rate
df["poverty_rate"] = df["poverty_below"] / df["poverty_total"]

# Rent burden (rent as share of monthly income)
df["rent_burden"] = (df["median_rent"] * 12) / df["median_income"]

# Population density requires area — placeholder for now
# (will be computed after spatial join with shapefile)

# ── SELECT FINAL COLUMNS ──────────────────────────────────────────────────────

final_cols = [
    "tract_id", "year",
    # Raw variables
    "population", "median_income", "median_rent", "median_home_value",
    # Derived features
    "share_college", "share_25_34",
    "homeownership_rate", "vacancy_rate",
    "share_pre1940",
    "share_white", "share_black", "share_hispanic",
    "poverty_rate", "rent_burden",
]

df_final = df[[c for c in final_cols if c in df.columns]].copy()

print(f"\nFinal dataset: {df_final.shape}")
print(df_final.head())

# ── SAVE OUTPUTS (paths depend on MODE) ───────────────────────────────────────

if MODE == "test_avg":
    # Average all feature columns across 2021-2024 per tract
    feature_cols = [c for c in final_cols if c not in ("tract_id", "year")]
    print("\nAveraging features across 2021-2024 per tract...")

    df_avg = (df_final
              .groupby("tract_id")[feature_cols]
              .mean()
              .reset_index())

    # Report how many years each tract had data for
    year_counts = df_final.groupby("tract_id")["year"].count().rename("years_available")
    df_avg = df_avg.merge(year_counts, on="tract_id", how="left")

    print(f"  Tracts with all 4 years:  {(df_avg['years_available'] == 4).sum():,}")
    print(f"  Tracts with 3 years:      {(df_avg['years_available'] == 3).sum():,}")
    print(f"  Tracts with fewer:        {(df_avg['years_available'] < 3).sum():,}")

    df_avg = df_avg.drop(columns="years_available")

    # Suffix all feature columns to make merge-ready
    df_out = (df_avg
              .set_index("tract_id")
              .add_suffix("_avg")
              .reset_index())

    df_out.to_csv("census_2021-2024_wide.csv", index=False)
    print(f"\nSaved census_2021-2024_wide.csv ({len(df_out):,} tracts, {df_out.shape[1]} columns)")
    print("Columns use '_avg' suffix — e.g. median_income_avg, share_college_avg")
    print("\nDone. Use census_2021-2024_wide.csv as your test set feature table.")

elif MODE == "multivintage":
    # Save long format — one row per tract per vintage year
    df_final.to_csv("census_multivintage.csv", index=False)
    print(f"\nSaved census_multivintage.csv ({len(df_final):,} rows, {df_final.shape[1]} columns)")
    print("Columns: tract_id, year, + all ACS features")
    print("Vintage years included:", sorted(df_final["year"].unique().tolist()))
    print("\nDone. Run build_all_datasets.py next to merge with Zillow and business density.")

elif MODE == "test":
    # Save long format
    df_final.to_csv("census_test_features.csv", index=False)
    print("\nSaved census_test_features.csv (long format)")

    # Save wide format — single year so just one suffix block
    df_2024 = (df_final[df_final["year"] == 2024]
               .drop(columns="year")
               .add_suffix("_2024")
               .rename(columns={"tract_id_2024": "tract_id"}))
    df_2024.to_csv("census_test_wide.csv", index=False)
    print(f"Saved census_test_wide.csv ({len(df_2024):,} tracts, {df_2024.shape[1]} columns)")
    print("\nDone. Use census_test_wide.csv as your test set feature table.")

else:
    # Save long format
    df_final.to_csv("census_features.csv", index=False)
    print("\nSaved census_features.csv (long format)")

    # Save wide format — all years side by side
    year_dfs = []
    for yr in YEARS:
        subset = (df_final[df_final["year"] == yr]
                  .drop(columns="year")
                  .add_suffix(f"_{yr}")
                  .rename(columns={f"tract_id_{yr}": "tract_id"}))
        year_dfs.append(subset)

    df_wide = year_dfs[0]
    for ydf in year_dfs[1:]:
        df_wide = df_wide.merge(ydf, on="tract_id", how="outer")

    print(f"Wide format: {df_wide.shape}")
    df_wide.to_csv("census_wide.csv", index=False)
    print("Saved census_wide.csv (wide format — one row per tract)")
    print("\nDone. Use census_wide.csv as your main feature table going forward.")
