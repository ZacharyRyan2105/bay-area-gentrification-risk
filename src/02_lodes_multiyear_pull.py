"""
lodes_multiyear_pull.py
------------------------
Downloads LEHD/LODES Workplace Area Characteristics (WAC) data for multiple
years and appends new rows to business_density.csv.

Pulls years: 2013, 2014, 2016, 2017, 2018, 2019
(2015 and 2020 are already in business_density.csv)

Each LODES WAC file is downloaded from:
    https://lehd.ces.census.gov/data/lodes/LODES8/ca/wac/ca_wac_S000_JT00_{YEAR}.csv.gz

Output:
    business_density.csv — updated in place with new year rows appended

Usage:
    python lodes_multiyear_pull.py
"""

import requests
import pandas as pd
import os
import gzip
import shutil
import struct

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

BAY_AREA_COUNTY_FIPS = ["06001", "06013", "06075", "06081", "06085"]
DBF_PATH             = "bay_area_tracts/bay_area_tracts.dbf"
OUTPUT_CSV           = "business_density.csv"

YEARS_TO_PULL = [2013, 2014, 2016, 2017, 2018, 2019]

SECTOR_COLS = {
    "CNS17": "jobs_arts",
    "CNS18": "jobs_food",
    "C000":  "jobs_total",
}

# ── TRACT AREAS ───────────────────────────────────────────────────────────────

def get_tract_areas():
    """Load tract area (sq mi) from DBF or tract_areas.csv fallback."""
    areas = None

    if os.path.exists(DBF_PATH):
        print("Reading tract areas from DBF...")
        with open(DBF_PATH, "rb") as f:
            header   = f.read(32)
            num_recs = struct.unpack("<I", header[4:8])[0]
            hdr_size = struct.unpack("<H", header[8:10])[0]
            rec_size = struct.unpack("<H", header[10:12])[0]
            fields = []
            while True:
                fd = f.read(32)
                if fd[0] == 0x0D:
                    break
                fields.append((fd[:11].replace(b"\x00", b"").decode("ascii"), fd[16]))
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
        df.columns = [c.replace("00","").replace("10","") for c in df.columns]
        if "CTIDFP" in df.columns:
            df = df.rename(columns={"CTIDFP": "tract_id"})
        df["area_sqmi"] = pd.to_numeric(df["ALAND"], errors="coerce") / 2_589_988
        df["tract_id"]  = df["tract_id"].astype(str).str.zfill(11)
        df["county_fips"] = df["tract_id"].str[:5]
        areas = df[df["county_fips"].isin(BAY_AREA_COUNTY_FIPS)][["tract_id","area_sqmi"]].copy()
        print(f"  {len(areas):,} tracts from DBF")

    if os.path.exists("tract_areas.csv"):
        gaz = pd.read_csv("tract_areas.csv", dtype={"tract_id": str})
        gaz["tract_id"] = gaz["tract_id"].astype(str).str.zfill(11)
        if areas is None:
            areas = gaz[["tract_id","area_sqmi"]]
        else:
            # Fill any gaps
            missing_ids = set(areas[areas["area_sqmi"].isna()]["tract_id"])
            fill = gaz[gaz["tract_id"].isin(missing_ids)][["tract_id","area_sqmi"]]
            areas = areas.set_index("tract_id")
            areas.update(fill.set_index("tract_id"))
            areas = areas.reset_index()

    if areas is None:
        raise FileNotFoundError(
            "No tract area source found. Need bay_area_tracts/bay_area_tracts.dbf "
            "or tract_areas.csv"
        )
    return areas

# ── DOWNLOAD ONE YEAR ─────────────────────────────────────────────────────────

def download_year(year):
    """Download and parse LODES WAC for one year. Returns tract-level DataFrame."""
    csv_path = f"ca_wac_{year}.csv"
    gz_path  = f"ca_wac_{year}.csv.gz"

    if not os.path.exists(csv_path):
        # Try LODES8 first, fall back to LODES7
        for lodes_ver in ["LODES8", "LODES7"]:
            url = (f"https://lehd.ces.census.gov/data/lodes/{lodes_ver}/ca/wac/"
                   f"ca_wac_S000_JT00_{year}.csv.gz")
            print(f"  Downloading {lodes_ver} {year}... ", end="", flush=True)
            resp = requests.get(url, timeout=120, stream=True)
            if resp.status_code == 200:
                with open(gz_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
                print("done")
                break
            else:
                print(f"HTTP {resp.status_code}", end=" ")
        else:
            print(f"\n  FAILED: Could not download {year} from LODES7 or LODES8")
            return None
    else:
        print(f"  Using cached {csv_path}")

    needed = ["w_geocode"] + list(SECTOR_COLS.keys())
    df = pd.read_csv(csv_path, usecols=needed, dtype={"w_geocode": str})
    df["tract_id"]    = df["w_geocode"].str.zfill(15).str[:11]
    df["county_fips"] = df["tract_id"].str[:5]
    df = df[df["county_fips"].isin(BAY_AREA_COUNTY_FIPS)].copy()
    return df

# ── MAIN ──────────────────────────────────────────────────────────────────────

tract_areas = get_tract_areas()

# Load existing business_density.csv to check which years already exist
existing = pd.read_csv(OUTPUT_CSV, dtype={"tract_id": str}) if os.path.exists(OUTPUT_CSV) else pd.DataFrame()
existing["tract_id"] = existing["tract_id"].astype(str).str.zfill(11)
existing_years = set(existing["year"].unique()) if "year" in existing.columns else set()
print(f"Existing years in {OUTPUT_CSV}: {sorted(existing_years)}")

new_rows = []

for year in YEARS_TO_PULL:
    if year in existing_years:
        print(f"  Year {year} already in {OUTPUT_CSV} — skipping")
        continue

    print(f"\nProcessing {year}...")
    raw = download_year(year)
    if raw is None:
        continue

    raw = raw.rename(columns=SECTOR_COLS)
    tract_df = raw.groupby("tract_id")[list(SECTOR_COLS.values())].sum().reset_index()
    tract_df = tract_df.merge(tract_areas, on="tract_id", how="left")

    # Compute density features
    tract_df["arts_density"]     = tract_df["jobs_arts"] / tract_df["area_sqmi"]
    tract_df["food_density"]     = tract_df["jobs_food"] / tract_df["area_sqmi"]
    tract_df["gentrify_density"] = (tract_df["jobs_arts"] + tract_df["jobs_food"]) / tract_df["area_sqmi"]
    tract_df["arts_job_share"]   = tract_df["jobs_arts"] / tract_df["jobs_total"].replace(0, pd.NA)
    tract_df["food_job_share"]   = tract_df["jobs_food"] / tract_df["jobs_total"].replace(0, pd.NA)
    tract_df["year"]             = year

    keep = ["tract_id", "year", "jobs_arts", "jobs_food", "jobs_total",
            "arts_density", "food_density", "gentrify_density",
            "arts_job_share", "food_job_share", "area_sqmi"]
    new_rows.append(tract_df[[c for c in keep if c in tract_df.columns]])

    print(f"  {len(tract_df):,} tracts  |  "
          f"mean food density: {tract_df['food_density'].mean():.1f} jobs/sqmi")

# ── APPEND AND SAVE ───────────────────────────────────────────────────────────

if new_rows:
    appended = pd.concat([existing] + new_rows, ignore_index=True)
    appended = appended.sort_values(["year", "tract_id"]).reset_index(drop=True)
    appended.to_csv(OUTPUT_CSV, index=False)
    years_now = sorted(appended["year"].unique().tolist())
    print(f"\nUpdated {OUTPUT_CSV}")
    print(f"Years now available: {years_now}")
    print(f"Total rows: {len(appended):,}")
else:
    print("\nNo new years to add — business_density.csv is already up to date.")

print("\nDone. Re-run build_multivintage_dataset.py to use exact LODES data for all vintages.")
print("Then update LODES_YEAR_MAP in build_multivintage_dataset.py to map each year to itself.")
