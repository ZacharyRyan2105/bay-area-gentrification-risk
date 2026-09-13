"""
standardize_permits.py
----------------------
Reads permit data from San Jose, San Francisco, and Oakland,
standardizes them into a single schema, geocodes addresses to
census tract IDs via the Census batch API, and saves a unified
permits dataset ready to merge with training/validation/test sets.

Inputs:
    sj_permits_clean.csv
    sf_new_construction_permits.csv
    Oakland New Residential Building Permits.xlsx - RecordList20260512 (1).csv
    Oakland 3+ Unit Res and Commercial Building Permits.xlsx - RecordList20260512 (2).csv

Outputs:
    permits_standardized.csv   — one row per permit, common schema, with tract_id
    permits_by_tract_year.csv  — aggregated features per tract per year (merge-ready)

Usage:
    pip install pandas requests
    python standardize_permits.py
"""

import pandas as pd
import numpy as np
import re
import requests
import io
import time
import os

# ── FILE PATHS ────────────────────────────────────────────────────────────────

PROJECT_DIR   = "/Users/zacharyryan/Documents/Claude/Projects/MS&E 125 Project"

SJ_PATH       = f"{PROJECT_DIR}/sj_permits_clean.csv"
SF_PATH       = f"{PROJECT_DIR}/sf_new_construction_permits.csv"
OAK_RES_PATH  = f"{PROJECT_DIR}/Oakland New Residential Building Permits.xlsx - RecordList20260512 (1).csv"
OAK_COM_PATH  = f"{PROJECT_DIR}/Oakland 3+ Unit Res and Commercial Building Permits.xlsx - RecordList20260512 (2).csv"
OUTPUT_STD    = f"{PROJECT_DIR}/permits_standardized.csv"
OUTPUT_AGG    = f"{PROJECT_DIR}/permits_by_tract_year.csv"
GEOCODE_CACHE = f"{PROJECT_DIR}/geocode_cache.csv"


# ── SHARED STATUS MAPPING ─────────────────────────────────────────────────────

# Maps raw status values → three canonical buckets
def map_status(raw):
    if pd.isna(raw):
        return "Unknown"
    r = str(raw).strip().lower()
    if any(k in r for k in ["complete", "final", "complet"]):
        return "Completed"
    if any(k in r for k in ["issued", "created", "routing", "in review",
                              "permit issued", "on hold", "reinstated",
                              "plan check", "approved"]):
        return "In Progress"
    if any(k in r for k in ["void", "cancel", "withdrawn", "expired",
                              "disapprove", "inactive"]):
        return "Void/Cancelled"
    return "Other"


# ── SHARED PERMIT TYPE MAPPING ────────────────────────────────────────────────

def map_permit_type(raw):
    if pd.isna(raw):
        return "Unknown"
    r = str(raw).strip().lower()
    # Catches: "new construction", "building new", "building/new", "building - new"
    if "new construction" in r or ("building" in r and "new" in r):
        return "New Construction"
    if "finish interior" in r or "tenant improvement" in r or "ti " in r:
        return "Tenant Improvement"
    if "addition" in r:
        return "Addition"
    if "demolition" in r or "demo" in r:
        return "Demolition"
    if "alteration" in r or "remodel" in r or "repair" in r:
        return "Alteration"
    if "foundation" in r:
        return "Foundation Only"
    if "shell" in r:
        return "Shell Only"
    return "Other"


# ── SHARED SUB-TYPE MAPPING ───────────────────────────────────────────────────

def map_sub_type(raw):
    if pd.isna(raw):
        return "Unknown"
    r = str(raw).strip().lower()
    if any(k in r for k in ["single family", "sfr", "sfd", "sfd ", "1 family",
                              "one family", "sfh"]):
        return "Single Family"
    if any(k in r for k in ["adu", "2nd unit", "accessory", "junior adu"]):
        return "ADU/2nd Unit"
    if any(k in r for k in ["condo", "condominium"]):
        return "Condo"
    if any(k in r for k in ["apartment", "multi", "multifamily", "3+ unit",
                              "3+res", "townhouse", "townhome"]):
        return "Multi-Family"
    if any(k in r for k in ["office", "medical", "dental", "clinic"]):
        return "Office/Medical"
    if any(k in r for k in ["retail", "commercial", "restaurant", "food",
                              "store", "shop"]):
        return "Retail/Commercial"
    if any(k in r for k in ["warehouse", "industrial", "manufacturing"]):
        return "Industrial/Warehouse"
    if any(k in r for k in ["mixed use", "mixed-use"]):
        return "Mixed Use"
    if any(k in r for k in ["hotel", "motel"]):
        return "Hotel/Motel"
    if any(k in r for k in ["garage", "parking", "carport"]):
        return "Parking/Garage"
    if any(k in r for k in ["school", "church", "hospital", "library",
                              "community", "civic"]):
        return "Civic/Institutional"
    return "Other"


# ── EXTRACT SQ FT FROM FREE TEXT ─────────────────────────────────────────────

SQFT_RE = re.compile(
    r'([\d,]+)\s*(?:sq\.?\s*ft\.?|sqft|square\s*feet|sf\b)',
    re.IGNORECASE
)

def extract_sqft(text):
    if pd.isna(text):
        return np.nan
    matches = [m.replace(',', '') for m in SQFT_RE.findall(str(text)) if m.replace(',', '').isdigit()]
    return float(max(int(m) for m in matches)) if matches else np.nan


# ── EXTRACT DWELL UNITS FROM FREE TEXT ───────────────────────────────────────

UNITS_RE = re.compile(
    r'(\d+)\s*(?:unit|dwelling|d\.u\.|du\b|residence|home|house|apartment)',
    re.IGNORECASE
)

def extract_units(text):
    if pd.isna(text):
        return np.nan
    matches = [m for m in UNITS_RE.findall(str(text)) if m.isdigit()]
    return float(max(int(m) for m in matches)) if matches else np.nan


# ── STANDARDIZE: SAN JOSE ─────────────────────────────────────────────────────

def load_sj():
    print("Loading San Jose permits...")
    df = pd.read_csv(SJ_PATH)

    # SJ status: infer from approvals column
    def sj_status(approvals):
        if pd.isna(approvals):
            return "Unknown"
        a = str(approvals).lower()
        if "complete" in a and "fnd only" not in a:
            return "Completed"
        if "fnd only" in a or "partial" in a:
            return "In Progress"
        return "In Progress"

    out = pd.DataFrame({
        "permit_id"    : df["folder_number"],
        "city"         : "San Jose",
        "date_filed"   : pd.to_datetime(df["permit_date"], errors="coerce"),
        "status"       : df["approvals"].apply(sj_status),
        "permit_type"  : df["work_type"].apply(map_permit_type),
        "sub_type"     : df["sub_type"].apply(map_sub_type),
        "address"      : df["address"].str.strip(),
        "valuation"    : pd.to_numeric(df["valuation"], errors="coerce"),
        "sq_ft"        : pd.to_numeric(df["sq_ft"], errors="coerce").replace(0, np.nan),
        "dwell_units"  : pd.to_numeric(df["dwell_units"], errors="coerce").replace(0, np.nan),
    })
    print(f"  {len(out):,} San Jose permits")
    return out


# ── STANDARDIZE: SAN FRANCISCO ───────────────────────────────────────────────

def load_sf():
    print("Loading San Francisco permits...")
    df = pd.read_csv(SF_PATH)

    # Reconstruct full address
    df["address"] = (
        df["street_number"].astype(str).str.strip() + " " +
        df["street_name"].astype(str).str.strip() + ", San Francisco, CA"
    )

    # SF sub_type comes from proposed_use
    out = pd.DataFrame({
        "permit_id"    : df["permit_number"].astype(str),
        "city"         : "San Francisco",
        "date_filed"   : pd.to_datetime(df["filed_date"], errors="coerce"),
        "status"       : df["status"].apply(map_status),
        "permit_type"  : df["permit_type_definition"].apply(map_permit_type),
        "sub_type"     : df["proposed_use"].apply(map_sub_type),
        "address"      : df["address"],
        "valuation"    : np.nan,     # not in this SF export
        "sq_ft"        : np.nan,     # not in this SF export
        "dwell_units"  : np.nan,
    })
    print(f"  {len(out):,} San Francisco permits")
    return out


# ── STANDARDIZE: OAKLAND ─────────────────────────────────────────────────────

def load_oakland():
    print("Loading Oakland permits...")
    res = pd.read_csv(OAK_RES_PATH)
    com = pd.read_csv(OAK_COM_PATH)

    res["source_type"] = "residential"
    com["source_type"] = "commercial_multifamily"
    df = pd.concat([res, com], ignore_index=True)

    # Sub-type: derive from description + record type
    def oak_sub_type(row):
        combined = " ".join([
            str(row.get("Description", "")),
            str(row.get("Record Type", ""))
        ])
        return map_sub_type(combined)

    out = pd.DataFrame({
        "permit_id"    : df["Record Number"],
        "city"         : "Oakland",
        "date_filed"   : pd.to_datetime(df["File Date"], errors="coerce"),
        "status"       : df["Status"].apply(map_status),
        "permit_type"  : df["Record Type"].apply(map_permit_type),
        "sub_type"     : df.apply(oak_sub_type, axis=1),
        "address"      : df["Address"].str.strip(),
        "valuation"    : np.nan,
        "sq_ft"        : df["Description"].apply(extract_sqft),
        "dwell_units"  : df["Description"].apply(extract_units),
    })
    print(f"  {len(out):,} Oakland permits")
    return out


# ── COMBINE ───────────────────────────────────────────────────────────────────

sj = load_sj()
sf = load_sf()
oak = load_oakland()

df = pd.concat([sj, sf, oak], ignore_index=True)
df = df[df["date_filed"].notna()]          # drop rows with no date
df = df[df["status"] != "Void/Cancelled"]  # drop voided permits
df["year_filed"] = df["date_filed"].dt.year

print(f"\nCombined: {len(df):,} permits ({df['city'].value_counts().to_dict()})")
print(f"Date range: {df['date_filed'].min().date()} → {df['date_filed'].max().date()}")


# ── GEOCODE TO CENSUS TRACT (Census Batch API) ───────────────────────────────
# The Census batch geocoder accepts up to 10,000 addresses per request.
# Returns state/county/tract GEOID for each address.

# GEOCODE_CACHE path defined at top of file
BATCH_SIZE    = 9500
GEOCODE_URL   = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"

def geocode_batch(address_df):
    """
    address_df: DataFrame with columns [id, address, city_str, state, zip]
    Returns dict: {id -> tract_id (11-digit GEOID)}
    """
    payload = address_df[["id","address","city_str","state","zip"]].copy()
    csv_str = payload.to_csv(index=False, header=False)

    try:
        resp = requests.post(
            GEOCODE_URL,
            data={
                "benchmark": "Public_AR_Current",
                "vintage":   "Census2010_Current",
            },
            files={"addressFile": ("addresses.csv", csv_str, "text/csv")},
            timeout=300,
        )
        if resp.status_code != 200:
            print(f"  Geocoder HTTP {resp.status_code}")
            return {}

        result = pd.read_csv(
            io.StringIO(resp.text),
            header=None,
            names=["id","input_addr","match","exact","matched_addr",
                   "coords","tiger_line","side","state_fips","county_fips",
                   "tract","block"],
            dtype=str,
        )
        result = result[result["match"] == "Match"]
        out = {}
        for _, row in result.iterrows():
            if pd.notna(row["state_fips"]) and pd.notna(row["county_fips"]) and pd.notna(row["tract"]):
                tract_id = (
                    str(row["state_fips"]).zfill(2) +
                    str(row["county_fips"]).zfill(3) +
                    str(row["tract"]).replace(".", "").zfill(6)
                )
                out[str(row["id"])] = tract_id
        return out

    except Exception as e:
        print(f"  Geocoding error: {e}")
        return {}


def geocode_all(df):
    """Geocode all unique addresses, using cache to avoid re-requesting."""
    # Load existing cache
    if os.path.exists(GEOCODE_CACHE):
        cache = pd.read_csv(GEOCODE_CACHE, dtype=str).set_index("address_key")["tract_id"].to_dict()
        print(f"  Loaded {len(cache):,} cached geocodes")
    else:
        cache = {}

    # Determine which addresses still need geocoding
    def address_key(row):
        return f"{row['address']}|{row['city']}"

    df["_addr_key"] = df.apply(address_key, axis=1)
    needs_geocoding = df[~df["_addr_key"].isin(cache)][["_addr_key","address","city"]].drop_duplicates("_addr_key")

    print(f"  {len(needs_geocoding):,} addresses to geocode ({len(cache):,} cached)")

    # Parse city into city_str / state / zip for the Census geocoder API
    def split_address(addr, city):
        try:
            parts = str(addr).split(",")
            street = parts[0].strip()
            if city == "Oakland" and len(parts) >= 2:
                city_state = parts[1].strip()
                tokens = city_state.split()
                # Expect format: "Oakland CA 94607"
                if len(tokens) >= 3 and tokens[-1].isdigit():
                    zip_code = tokens[-1]
                    state    = tokens[-2]
                    city_str = " ".join(tokens[:-2])
                elif len(tokens) >= 2 and tokens[-1].isdigit():
                    zip_code = tokens[-1]
                    state    = "CA"
                    city_str = tokens[0]
                else:
                    zip_code = ""
                    state    = "CA"
                    city_str = "Oakland"
            elif city == "San Francisco":
                city_str, state, zip_code = "San Francisco", "CA", ""
            else:
                city_str, state, zip_code = "San Jose", "CA", ""
            return street, city_str, state, zip_code
        except Exception:
            # Fallback: just send the raw first segment with city defaults
            return str(addr).split(",")[0].strip(), city, "CA", ""

    new_results = {}
    batches = [needs_geocoding.iloc[i:i+BATCH_SIZE]
               for i in range(0, len(needs_geocoding), BATCH_SIZE)]

    for b_num, batch in enumerate(batches, 1):
        print(f"  Geocoding batch {b_num}/{len(batches)} ({len(batch)} addresses)...")
        rows = []
        for idx, row in batch.iterrows():
            street, city_str, state, zip_code = split_address(row["address"], row["city"])
            rows.append({
                "id":        str(idx),
                "address":   street,
                "city_str":  city_str,
                "state":     state,
                "zip":       zip_code,
                "_addr_key": row["_addr_key"],
            })
        batch_df = pd.DataFrame(rows)
        id_to_key = batch_df.set_index("id")["_addr_key"].to_dict()

        results = geocode_batch(batch_df)
        for id_, tract_id in results.items():
            key = id_to_key.get(id_)
            if key:
                new_results[key] = tract_id

        # Save cache after every batch so progress isn't lost on crash
        cache.update(new_results)
        cache_df = pd.DataFrame(list(cache.items()), columns=["address_key","tract_id"])
        cache_df.to_csv(GEOCODE_CACHE, index=False)
        print(f"    Batch {b_num} done — {len(results):,} matched. Cache: {len(cache):,} total.")

        time.sleep(1)   # be polite

    print(f"  Geocoded {len(new_results):,} new addresses. Cache now {len(cache):,} entries.")

    # Map back to permit rows
    df["tract_id"] = df["_addr_key"].map(cache)
    df = df.drop(columns=["_addr_key"])
    return df


print("\nGeocoding addresses to census tract IDs...")
df = geocode_all(df)

matched = df["tract_id"].notna().sum()
print(f"  Tract IDs assigned: {matched:,} of {len(df):,} ({matched/len(df):.1%})")


# ── SAVE STANDARDIZED FILE ────────────────────────────────────────────────────

col_order = [
    "permit_id", "city", "date_filed", "year_filed",
    "status", "permit_type", "sub_type",
    "address", "tract_id",
    "valuation", "sq_ft", "dwell_units",
]
df = df[[c for c in col_order if c in df.columns]]
df.to_csv(OUTPUT_STD, index=False)
print(f"\nSaved {OUTPUT_STD}  ({len(df):,} rows, {df.shape[1]} columns)")


# ── AGGREGATE BY TRACT × YEAR ─────────────────────────────────────────────────
# Creates merge-ready features: for each tract, count permit activity
# in each calendar year. You can then join to your training/validation/test
# sets by tract_id and a date window.

df_geo = df[df["tract_id"].notna()].copy()

# New construction only (most relevant for gentrification signal)
nc = df_geo[df_geo["permit_type"] == "New Construction"].copy()
all_p = df_geo.copy()

def agg_permits(grp_df, prefix):
    g = grp_df.groupby(["tract_id", "year_filed"])
    return pd.DataFrame({
        f"{prefix}_count"        : g["permit_id"].count(),
        f"{prefix}_valuation_sum": g["valuation"].sum(),
        f"{prefix}_valuation_avg": g["valuation"].mean(),
        f"{prefix}_sqft_sum"     : g["sq_ft"].sum(),
        f"{prefix}_units_sum"    : g["dwell_units"].sum(),
    }).reset_index()

nc_agg  = agg_permits(nc,    "new_const")
all_agg = agg_permits(all_p, "all_permits")

agg = nc_agg.merge(all_agg, on=["tract_id","year_filed"], how="outer")
agg = agg.sort_values(["tract_id","year_filed"]).reset_index(drop=True)
agg.to_csv(OUTPUT_AGG, index=False)

print(f"Saved {OUTPUT_AGG}  ({len(agg):,} tract-year rows)")
print(f"\nSample aggregated output:")
print(agg.head(8).to_string())
print(f"\nTo merge with your training set:")
print("  train = train.merge(agg[agg['year_filed'] == 2010], on='tract_id', how='left')")
print("  val   = val.merge(agg[agg['year_filed'] == 2015], on='tract_id', how='left')")
print("  test  = test.merge(agg[agg['year_filed'] == 2023], on='tract_id', how='left')")
