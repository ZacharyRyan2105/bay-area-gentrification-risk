"""
walkscore_pull.py
-----------------
Fetches Walk Score, Transit Score, and Bike Score for each Bay Area
census tract centroid. Results are used as walkability/transit features
in the gentrification prediction model.

Usage:
    python walkscore_pull.py

Requirements:
    pip install requests pandas geopandas

Before running:
    Paste your Walk Score API key into WALKSCORE_API_KEY below.
    Sign up free at: https://www.walkscore.com/professional/api.php

Output:
    walkscore_features.csv — one row per tract with walk/transit/bike scores
"""

import os
import json
import time
import requests
import pandas as pd
import geopandas as gpd

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

WALKSCORE_API_KEY = os.environ.get("WALKSCORE_API_KEY", "")

SHAPEFILE_PATH = "bay_area_tracts/bay_area_tracts.shp"

BAY_AREA_COUNTIES = ["06075", "06001", "06013", "06081", "06085"]

CACHE_DIR = "walkscore_cache"
OUTPUT_CSV = "walkscore_features.csv"

# Walk Score API endpoint
WS_URL = "https://api.walkscore.com/score"

# ── SETUP ─────────────────────────────────────────────────────────────────────

os.makedirs(CACHE_DIR, exist_ok=True)

# ── LOAD TRACTS ───────────────────────────────────────────────────────────────

print("Loading census tract shapefile...")
tracts = gpd.read_file(SHAPEFILE_PATH)

# Normalize column names (handles 2000-vintage TIGER naming like STATEFP00)
tracts.columns = [c.replace("00", "").replace("10", "") for c in tracts.columns]

if "CTIDFP" in tracts.columns:
    tracts = tracts.rename(columns={"CTIDFP": "GEOID"})

# Build GEOID if not present
if "GEOID" not in tracts.columns:
    tracts["GEOID"] = (
        tracts["STATEFP"].astype(str).str.zfill(2)
        + tracts["COUNTYFP"].astype(str).str.zfill(3)
        + tracts["TRACTCE"].astype(str).str.zfill(6)
    )

# Filter to Bay Area
tracts["county_fips"] = tracts["GEOID"].astype(str).str[:5]
tracts = tracts[tracts["county_fips"].isin(BAY_AREA_COUNTIES)].copy()

# Reproject to WGS84 for lat/lon
tracts = tracts.to_crs("EPSG:4326")
tracts["lat"] = tracts.geometry.centroid.y
tracts["lon"] = tracts.geometry.centroid.x
tracts["tract_id"] = tracts["GEOID"].astype(str).str.zfill(11)

print(f"  Loaded {len(tracts)} Bay Area census tracts.")

# ── FETCH FUNCTION ────────────────────────────────────────────────────────────

def fetch_walkscore(tract_id, lat, lon):
    """
    Fetches Walk Score, Transit Score, and Bike Score for a lat/lon.
    Caches results locally to survive reruns.
    Returns a dict with the three scores (None if API error).
    """
    cache_file = os.path.join(CACHE_DIR, f"{tract_id}.json")

    # Return cached result if available
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    params = {
        "format":    "json",
        "lat":       lat,
        "lon":       lon,
        "transit":   1,
        "bike":      1,
        "wsapikey":  WALKSCORE_API_KEY,
    }

    try:
        resp = requests.get(WS_URL, params=params, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            result = {
                "tract_id":      tract_id,
                "walk_score":    data.get("walkscore"),
                "walk_desc":     data.get("description"),
                "transit_score": data.get("transit", {}).get("score") if data.get("transit") else None,
                "transit_desc":  data.get("transit", {}).get("description") if data.get("transit") else None,
                "bike_score":    data.get("bike", {}).get("score") if data.get("bike") else None,
                "bike_desc":     data.get("bike", {}).get("description") if data.get("bike") else None,
            }
            # Cache the result
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result

        elif resp.status_code == 401:
            print(f"\n  ERROR: Invalid API key. Check WALKSCORE_API_KEY.")
            return None

        elif resp.status_code == 429:
            print(f"\n  Rate limit hit. Pausing 60 seconds...")
            time.sleep(60)
            return fetch_walkscore(tract_id, lat, lon)

        else:
            print(f"\n  API error {resp.status_code} for tract {tract_id}")
            return None

    except Exception as e:
        print(f"\n  Request failed for tract {tract_id}: {e}")
        return None


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

print(f"\nFetching Walk Scores for {len(tracts)} tracts...")
print("(Already-cached tracts are skipped instantly)\n")

results = []
total = len(tracts)

for i, row in enumerate(tracts.itertuples(), 1):
    cache_file = os.path.join(CACHE_DIR, f"{row.tract_id}.json")
    cached = os.path.exists(cache_file)

    result = fetch_walkscore(row.tract_id, row.lat, row.lon)

    if result:
        results.append(result)

    # Progress update every 25 tracts
    if i % 25 == 0 or i == total:
        done = len([f for f in os.listdir(CACHE_DIR) if f.endswith(".json")])
        print(f"  {i}/{total} tracts processed  ({done} cached total)")

    # Polite delay for non-cached requests only
    if not cached:
        time.sleep(0.1)

# ── SAVE OUTPUT ───────────────────────────────────────────────────────────────

print("\nSaving results...")
df = pd.DataFrame(results)

# Reorder columns
col_order = [
    "tract_id",
    "walk_score", "walk_desc",
    "transit_score", "transit_desc",
    "bike_score", "bike_desc",
]
df = df[[c for c in col_order if c in df.columns]]

df.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone. {len(df)} tracts saved to {OUTPUT_CSV}")
print(f"\nScore summary:")
print(f"  Walk Score    — mean: {df['walk_score'].mean():.1f},  min: {df['walk_score'].min()},  max: {df['walk_score'].max()}")
if "transit_score" in df.columns:
    print(f"  Transit Score — mean: {df['transit_score'].mean():.1f},  min: {df['transit_score'].min()},  max: {df['transit_score'].max()}")
if "bike_score" in df.columns:
    print(f"  Bike Score    — mean: {df['bike_score'].mean():.1f},  min: {df['bike_score'].min()},  max: {df['bike_score'].max()}")

print("\nNext step: merge walkscore_features.csv into your feature matrix on tract_id.")
