"""
business_density.py
-------------------
Downloads LEHD/LODES Workplace Area Characteristics (WAC) data for
California and computes arts/food-service job density per square mile
at the census tract level for 2010, 2015, 2020, and 2022.

This serves as the historical proxy for gentrification-signal business
density (coffee shops, galleries, yoga studios, wine bars) that Yelp
only provides for 2026.

Usage:
    python business_density.py

Requirements:
    pip install requests pandas
    (No geopandas needed — reads tract areas directly from the DBF file)

Output:
    business_density.csv      — one row per tract per year with density features
    business_density_wide.csv — one row per tract with year-suffixed columns
"""

import requests
import pandas as pd
import os
import gzip
import shutil
import struct

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

YEARS = [2010, 2015, 2020, 2022, 2023]

# Bay Area county FIPS codes (5-digit: state + county)
BAY_AREA_COUNTY_FIPS = ["06001", "06013", "06075", "06081", "06085"]

# Path to the .dbf file inside your bay_area_tracts folder
DBF_PATH = "bay_area_tracts/bay_area_tracts.dbf"

# LODES WAC file URL pattern (LODES8, all workers, all jobs)
LODES_URL = "https://lehd.ces.census.gov/data/lodes/LODES8/ca/wac/ca_wac_S000_JT00_{year}.csv.gz"

# ── INDUSTRY SECTORS OF INTEREST ──────────────────────────────────────────────
# CNS17 — Arts, Entertainment, and Recreation
#          (galleries, music venues, fitness studios, yoga)
# CNS18 — Accommodation and Food Services
#          (cafes, restaurants, bars, wine bars, specialty food)
# C000  — Total jobs (used for normalization)

SECTOR_COLS = {
    "CNS17": "jobs_arts",
    "CNS18": "jobs_food",
    "C000":  "jobs_total",
}

# ── READ TRACT AREAS FROM DBF (no geopandas needed) ──────────────────────────

def get_tract_areas(dbf_path):
    """
    Reads GEOID and precomputed land area (ALAND) from the shapefile's .dbf.
    TIGER/Line shapefiles include ALAND in square meters — we convert to sq miles.
    Returns a DataFrame with tract_id and area_sqmi.
    """
    print("Reading tract areas from DBF file...")

    with open(dbf_path, "rb") as f:
        header = f.read(32)
        num_records = struct.unpack("<I", header[4:8])[0]
        header_size = struct.unpack("<H", header[8:10])[0]
        record_size = struct.unpack("<H", header[10:12])[0]

        # Parse field descriptors
        fields = []
        while True:
            fd = f.read(32)
            if fd[0] == 0x0D:
                break
            name = fd[:11].replace(b"\x00", b"").decode("ascii")
            ftype = chr(fd[11])
            flen = fd[16]
            fields.append((name, ftype, flen))

        # Seek to first record
        f.seek(header_size)
        records = []
        for _ in range(num_records):
            raw = f.read(record_size)
            row = {}
            pos = 1
            for name, ftype, flen in fields:
                val = raw[pos:pos + flen].decode("latin-1").strip()
                row[name] = val
                pos += flen
            records.append(row)

    df = pd.DataFrame(records)

    # Normalize column names (handles STATEFP00, CTIDFP00 vintage naming)
    df.columns = [c.replace("00", "").replace("10", "") for c in df.columns]

    # CTIDFP is the full 11-digit tract GEOID
    if "CTIDFP" in df.columns:
        df = df.rename(columns={"CTIDFP": "tract_id"})

    # ALAND is land area in square meters (precomputed by Census Bureau)
    df["area_sqmi"] = pd.to_numeric(df["ALAND"], errors="coerce") / 2_589_988

    df["tract_id"] = df["tract_id"].astype(str).str.zfill(11)

    # Filter to Bay Area
    df["county_fips"] = df["tract_id"].str[:5]
    df = df[df["county_fips"].isin(BAY_AREA_COUNTY_FIPS)]

    print(f"  {len(df)} Bay Area tracts loaded.")
    return df[["tract_id", "area_sqmi"]].copy()


# ── DOWNLOAD LODES FILES ──────────────────────────────────────────────────────

def download_lodes(year):
    """
    Downloads the California LODES WAC file for a given year.
    Caches the decompressed CSV locally — safe to rerun.
    Returns a DataFrame filtered to Bay Area tracts, or None on failure.
    """
    url = LODES_URL.format(year=year)
    gz_path = f"ca_wac_{year}.csv.gz"
    csv_path = f"ca_wac_{year}.csv"

    if not os.path.exists(csv_path):
        print(f"  Downloading {year} LODES file (~50MB)...")
        try:
            resp = requests.get(url, timeout=120, stream=True)
            if resp.status_code != 200:
                print(f"  Failed to download {year}: HTTP {resp.status_code}")
                return None
            with open(gz_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            with gzip.open(gz_path, "rb") as f_in:
                with open(csv_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
            print(f"  Downloaded and decompressed.")
        except Exception as e:
            print(f"  Download error for {year}: {e}")
            return None
    else:
        print(f"  Using cached {year} file.")

    # Load only needed columns
    needed_cols = ["w_geocode"] + list(SECTOR_COLS.keys())
    df = pd.read_csv(csv_path, usecols=needed_cols, dtype={"w_geocode": str})

    # w_geocode is a 15-digit census block GEOID; first 11 digits = tract
    df["tract_id"] = df["w_geocode"].str.zfill(15).str[:11]
    df["county_fips"] = df["tract_id"].str[:5]
    df = df[df["county_fips"].isin(BAY_AREA_COUNTY_FIPS)].copy()

    print(f"  Bay Area records: {len(df):,}")
    return df


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

tract_areas = get_tract_areas(DBF_PATH)
all_dfs = []

for year in YEARS:
    print(f"\nProcessing {year}...")
    df = download_lodes(year)
    if df is None:
        print(f"  Skipping {year}.")
        continue

    # Rename sector columns
    df = df.rename(columns=SECTOR_COLS)

    # Aggregate from census block to census tract
    tract_df = df.groupby("tract_id")[list(SECTOR_COLS.values())].sum().reset_index()

    # Merge tract areas
    tract_df = tract_df.merge(tract_areas, on="tract_id", how="left")

    # Density per square mile
    tract_df["arts_density"]     = tract_df["jobs_arts"]  / tract_df["area_sqmi"]
    tract_df["food_density"]     = tract_df["jobs_food"]  / tract_df["area_sqmi"]
    tract_df["gentrify_density"] = (tract_df["jobs_arts"] + tract_df["jobs_food"]) / tract_df["area_sqmi"]

    # Share of total jobs
    tract_df["arts_job_share"]   = tract_df["jobs_arts"] / tract_df["jobs_total"].replace(0, pd.NA)
    tract_df["food_job_share"]   = tract_df["jobs_food"] / tract_df["jobs_total"].replace(0, pd.NA)

    tract_df["year"] = year
    all_dfs.append(tract_df)

    print(f"  Tracts with data: {len(tract_df)}")
    print(f"  Avg arts density: {tract_df['arts_density'].mean():.2f} jobs/sqmi")
    print(f"  Avg food density: {tract_df['food_density'].mean():.2f} jobs/sqmi")

# ── COMBINE AND SAVE ──────────────────────────────────────────────────────────

print("\nCombining all years...")
df_all = pd.concat(all_dfs, ignore_index=True)

final_cols = [
    "tract_id", "year",
    "jobs_arts", "jobs_food", "jobs_total",
    "arts_density", "food_density", "gentrify_density",
    "arts_job_share", "food_job_share",
    "area_sqmi",
]
df_final = df_all[final_cols].copy()

# Long format
df_final.to_csv("business_density.csv", index=False)
print(f"Saved business_density.csv ({len(df_final)} rows)")

# Wide format — one row per tract, columns suffixed by year
year_dfs = []
for year in YEARS:
    subset = df_final[df_final["year"] == year].drop(columns="year")
    subset = subset.add_suffix(f"_{year}").rename(columns={f"tract_id_{year}": "tract_id"})
    year_dfs.append(subset)

df_wide = year_dfs[0]
for ydf in year_dfs[1:]:
    df_wide = df_wide.merge(ydf, on="tract_id", how="outer")

df_wide.to_csv("business_density_wide.csv", index=False)
print(f"Saved business_density_wide.csv ({len(df_wide)} tracts)")

print("\nDone. Key columns for your feature matrix:")
print("  arts_density_YYYY     — Arts/Entertainment jobs per square mile")
print("  food_density_YYYY     — Food/Accommodation jobs per square mile")
print("  gentrify_density_YYYY — Combined arts + food jobs per square mile")
print("  arts_job_share_YYYY   — Arts jobs as share of all local jobs")
print("  food_job_share_YYYY   — Food jobs as share of all local jobs")
