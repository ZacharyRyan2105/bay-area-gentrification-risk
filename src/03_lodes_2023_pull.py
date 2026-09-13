"""
lodes_2023_pull.py
------------------
Downloads LEHD/LODES 2023 Workplace Area Characteristics (WAC) data
for California and computes arts/food-service job density per square
mile at the census tract level for Bay Area tracts.

Usage:
    python lodes_2023_pull.py

Requirements:
    pip install requests pandas

Output:
    lodes_2023_wide.csv — one row per tract with 2023 density features
"""

import requests
import pandas as pd
import os
import gzip
import shutil
import struct
import io

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

BAY_AREA_COUNTY_FIPS = ["06001", "06013", "06075", "06081", "06085"]
DBF_PATH   = "bay_area_tracts/bay_area_tracts.dbf"
LODES_URL  = "https://lehd.ces.census.gov/data/lodes/LODES8/ca/wac/ca_wac_S000_JT00_2023.csv.gz"
GZ_PATH    = "ca_wac_2023.csv.gz"
CSV_PATH   = "ca_wac_2023.csv"
OUTPUT_CSV = "lodes_2023_wide.csv"

SECTOR_COLS = {
    "CNS17": "jobs_arts",
    "CNS18": "jobs_food",
    "C000":  "jobs_total",
}

# ── READ TRACT AREAS FROM DBF ─────────────────────────────────────────────────

def get_tract_areas(dbf_path):
    print("Reading tract areas from DBF...")
    with open(dbf_path, "rb") as f:
        header     = f.read(32)
        num_recs   = struct.unpack("<I", header[4:8])[0]
        hdr_size   = struct.unpack("<H", header[8:10])[0]
        rec_size   = struct.unpack("<H", header[10:12])[0]
        fields = []
        while True:
            fd = f.read(32)
            if fd[0] == 0x0D:
                break
            fields.append((fd[:11].replace(b"\x00", b"").decode("ascii"),
                           fd[16]))
        f.seek(hdr_size)
        records = []
        for _ in range(num_recs):
            raw = f.read(rec_size)
            row, pos = {}, 1
            for name, flen in fields:
                row[name] = raw[pos:pos+flen].decode("latin-1").strip()
                pos += flen
            records.append(row)

    df = pd.DataFrame(records)
    df.columns = [c.replace("00", "").replace("10", "") for c in df.columns]
    if "CTIDFP" in df.columns:
        df = df.rename(columns={"CTIDFP": "tract_id"})
    df["area_sqmi"] = pd.to_numeric(df["ALAND"], errors="coerce") / 2_589_988
    df["tract_id"]  = df["tract_id"].astype(str).str.zfill(11)
    df["county_fips"] = df["tract_id"].str[:5]
    df = df[df["county_fips"].isin(BAY_AREA_COUNTY_FIPS)]
    print(f"  {len(df):,} Bay Area tracts loaded.")
    return df[["tract_id", "area_sqmi"]].copy()

# ── DOWNLOAD LODES 2023 ───────────────────────────────────────────────────────

def download_lodes():
    if not os.path.exists(CSV_PATH):
        print("Downloading 2023 LODES file (~50MB)...")
        resp = requests.get(LODES_URL, timeout=120, stream=True)
        if resp.status_code != 200:
            print(f"  Failed: HTTP {resp.status_code}")
            print("  2023 LODES data may not be published yet. Try 2022 instead.")
            return None
        with open(GZ_PATH, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        with gzip.open(GZ_PATH, "rb") as f_in:
            with open(CSV_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(GZ_PATH)
        print("  Downloaded and decompressed.")
    else:
        print("  Using cached 2023 file.")

    needed_cols = ["w_geocode"] + list(SECTOR_COLS.keys())
    df = pd.read_csv(CSV_PATH, usecols=needed_cols, dtype={"w_geocode": str})
    df["tract_id"]    = df["w_geocode"].str.zfill(15).str[:11]
    df["county_fips"] = df["tract_id"].str[:5]
    df = df[df["county_fips"].isin(BAY_AREA_COUNTY_FIPS)].copy()
    print(f"  Bay Area records: {len(df):,}")
    return df

# ── MAIN ──────────────────────────────────────────────────────────────────────

tract_areas = get_tract_areas(DBF_PATH)

print("\nDownloading LODES 2023...")
df = download_lodes()
if df is None:
    exit()

# Rename and aggregate block → tract
df = df.rename(columns=SECTOR_COLS)
tract_df = df.groupby("tract_id")[list(SECTOR_COLS.values())].sum().reset_index()

# Merge tract areas (DBF first, then Gazetteer backfill if available)
tract_df = tract_df.merge(tract_areas, on="tract_id", how="left")

# Backfill from tract_areas.csv if it exists (from get_tract_areas.py)
if os.path.exists("tract_areas.csv"):
    gaz = pd.read_csv("tract_areas.csv", dtype={"tract_id": str})
    gaz["tract_id"] = gaz["tract_id"].astype(str).str.zfill(11)
    gaz = gaz.rename(columns={"area_sqmi": "area_sqmi_gaz"})
    tract_df = tract_df.merge(gaz, on="tract_id", how="left")
    missing = tract_df["area_sqmi"].isna()
    tract_df.loc[missing, "area_sqmi"] = tract_df.loc[missing, "area_sqmi_gaz"]
    tract_df = tract_df.drop(columns="area_sqmi_gaz")

# Compute density features
tract_df["arts_density_2023"]     = tract_df["jobs_arts"] / tract_df["area_sqmi"]
tract_df["food_density_2023"]     = tract_df["jobs_food"] / tract_df["area_sqmi"]
tract_df["gentrify_density_2023"] = (tract_df["jobs_arts"] + tract_df["jobs_food"]) / tract_df["area_sqmi"]
tract_df["arts_job_share_2023"]   = tract_df["jobs_arts"] / tract_df["jobs_total"].replace(0, pd.NA)
tract_df["food_job_share_2023"]   = tract_df["jobs_food"] / tract_df["jobs_total"].replace(0, pd.NA)

# Rename raw count columns to include year suffix
tract_df = tract_df.rename(columns={
    "jobs_arts":  "jobs_arts_2023",
    "jobs_food":  "jobs_food_2023",
    "jobs_total": "jobs_total_2023",
    "area_sqmi":  "area_sqmi_2023",
})

# Save
tract_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {OUTPUT_CSV} ({len(tract_df):,} tracts, {tract_df.shape[1]} columns)")
print(f"  Tracts with valid density: {tract_df['arts_density_2023'].notna().sum():,}")
print(f"  Avg arts density:  {tract_df['arts_density_2023'].mean():.2f} jobs/sqmi")
print(f"  Avg food density:  {tract_df['food_density_2023'].mean():.2f} jobs/sqmi")
print("\nDone. Merge lodes_2023_wide.csv into your test set on tract_id.")
