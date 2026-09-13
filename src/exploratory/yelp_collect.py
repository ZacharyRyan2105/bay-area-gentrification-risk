"""
yelp_collect.py
---------------
Collects Yelp business listings for gentrification-related categories
across Bay Area census tracts.

Usage:
    python yelp_collect.py

Requirements:
    pip install requests geopandas pandas shapely

Before running:
    1. Export your API key:  export YELP_API_KEY=your_key_here
    2. Place the Bay Area TIGER census tract shapefile in the same folder
       as this script, or update SHAPEFILE_PATH to point to it.
       Download from: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
       (Select year 2010 → Census Tracts → California, then filter to Bay Area counties)

Output:
    yelp_businesses.csv  — one row per business with name, category,
                           lat/lon, earliest review date, and census tract GEOID
"""

import os
import time
import json
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

YELP_API_KEY = os.environ.get("YELP_API_KEY", "")

# Path to the Bay Area census tract shapefile (from TIGER/Line 2010)
SHAPEFILE_PATH = "bay_area_tracts/bay_area_tracts.shp"

# Business categories linked to early gentrification signals
# Passed as a single comma-separated string to use 1 request per tract
# instead of 7, keeping total requests within the 300/day rate limit.
CATEGORIES_QUERY = "cafes,galleries,wine_bars,yoga"
CATEGORIES = CATEGORIES_QUERY.split(",")  # Used for labeling only

# Bay Area county FIPS codes (used to filter the shapefile)
BAY_AREA_COUNTIES = ["06075", "06001", "06013", "06081", "06085"]
#                    SF       Alameda  Contra   San     Santa
#                                     Costa    Mateo   Clara

# Search radius around each tract centroid in meters (roughly 1 mile)
SEARCH_RADIUS = 750

# Yelp returns max 50 results per request; max offset is 1000
RESULTS_PER_REQUEST = 50
MAX_OFFSET = 950  # Stay under Yelp's 1000-result cap per search

# Cache folder — saves raw API responses so you never re-query the same search
CACHE_DIR = "yelp_cache"

# Output file
OUTPUT_CSV = "yelp_businesses.csv"

# ── SETUP ─────────────────────────────────────────────────────────────────────

os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {
    "Authorization": f"Bearer {YELP_API_KEY}",
    "Accept": "application/json",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def cache_path(lat, lon, category, offset):
    """Returns a local file path for caching a specific API call."""
    # Category excluded from key since we now always use the same combined query.
    # Also checks for legacy single-category cache files from earlier runs.
    combined_key = f"{lat:.4f}_{lon:.4f}_{offset}"
    combined_path = os.path.join(CACHE_DIR, f"{combined_key}.json")
    if os.path.exists(combined_path):
        return combined_path
    # Fall back to legacy format (individual category name)
    legacy_key = f"{lat:.4f}_{lon:.4f}_{category}_{offset}"
    legacy_path = os.path.join(CACHE_DIR, f"{legacy_key}.json")
    if os.path.exists(legacy_path):
        return legacy_path
    # Default to new combined format for new writes
    return combined_path


def fetch_businesses(lat, lon, category, offset=0):
    """
    Calls the Yelp Business Search endpoint.
    Returns a list of business dicts, or [] on error.
    Caches results locally to avoid re-querying on reruns.
    """
    path = cache_path(lat, lon, category, offset)

    # Return cached result if it exists (no sleep needed)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f), True  # True = was cached

    params = {
        "latitude": lat,
        "longitude": lon,
        "categories": category,
        "radius": SEARCH_RADIUS,
        "limit": RESULTS_PER_REQUEST,
        "offset": offset,
    }

    try:
        resp = requests.get(
            "https://api.yelp.com/v3/businesses/search",
            headers=HEADERS,
            params=params,
            timeout=10,
        )

        if resp.status_code == 200:
            businesses = resp.json().get("businesses", [])
            # Cache the result
            with open(path, "w") as f:
                json.dump(businesses, f)
            return businesses, False  # False = was not cached

        elif resp.status_code == 429:
            # Rate limit hit — pause and retry once
            print("  Rate limit hit, pausing 60 seconds...")
            time.sleep(60)
            return fetch_businesses(lat, lon, category, offset)

        else:
            print(f"  API error {resp.status_code} for {category} at ({lat:.4f}, {lon:.4f})")
            return [], False

    except requests.exceptions.RequestException as e:
        print(f"  Request failed: {e}")
        return [], False


def get_earliest_review_date(business_id):
    """
    Fetches the earliest review date for a business as a proxy
    for its opening date (Yelp doesn't expose opening date directly).
    Returns a date string or None.
    """
    path = os.path.join(CACHE_DIR, f"reviews_{business_id}.json")

    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    try:
        resp = requests.get(
            f"https://api.yelp.com/v3/businesses/{business_id}/reviews",
            headers=HEADERS,
            timeout=10,
        )
        if resp.status_code == 200:
            reviews = resp.json().get("reviews", [])
            if reviews:
                dates = [r["time_created"] for r in reviews if "time_created" in r]
                earliest = min(dates) if dates else None
                with open(path, "w") as f:
                    json.dump(earliest, f)
                return earliest
        return None
    except:
        return None


# ── LOAD CENSUS TRACTS ────────────────────────────────────────────────────────

print("Loading census tract shapefile...")
tracts = gpd.read_file(SHAPEFILE_PATH)

# Normalize column names — handle both 2000-vintage (STATEFP00) and
# 2010-vintage (STATEFP10 / STATEFP) TIGER column naming conventions
tracts.columns = [c.replace("00", "").replace("10", "") for c in tracts.columns]

# CTIDFP is the full 11-digit GEOID in 2000-vintage files
if "CTIDFP" in tracts.columns:
    tracts = tracts.rename(columns={"CTIDFP": "GEOID"})

# Filter to Bay Area counties using the first 5 digits of GEOID
if "GEOID" in tracts.columns:
    tracts["county_fips"] = tracts["GEOID"].astype(str).str[:5]
    tracts = tracts[tracts["county_fips"].isin(BAY_AREA_COUNTIES)]
elif "COUNTYFP" in tracts.columns:
    tracts["county_fips"] = "06" + tracts["COUNTYFP"].astype(str).str.zfill(3)
    tracts = tracts[tracts["county_fips"].isin(BAY_AREA_COUNTIES)]

# Ensure GEOID column exists
if "GEOID" not in tracts.columns:
    tracts["GEOID"] = (
        tracts["STATEFP"].astype(str).str.zfill(2)
        + tracts["COUNTYFP"].astype(str).str.zfill(3)
        + tracts["TRACTCE"].astype(str).str.zfill(6)
    )

# Reproject to WGS84 (lat/lon) for Yelp API calls
tracts = tracts.to_crs("EPSG:4326")

# Compute centroid of each tract
tracts["centroid"] = tracts.geometry.centroid
tracts["lat"] = tracts["centroid"].y
tracts["lon"] = tracts["centroid"].x

print(f"  Loaded {len(tracts)} census tracts across Bay Area counties.")

# ── LOAD FROM CACHE ───────────────────────────────────────────────────────────
# All tracts have been collected. Read every JSON file in the cache directory
# directly — no API calls needed.

print("Loading all businesses from cache...")
all_records = []

cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json") and not f.startswith("reviews_")]
print(f"  Found {len(cache_files)} cache files.")

for fname in cache_files:
    fpath = os.path.join(CACHE_DIR, fname)
    try:
        with open(fpath) as f:
            businesses = json.load(f)
        if not isinstance(businesses, list):
            continue
        for biz in businesses:
            biz_lat = biz.get("coordinates", {}).get("latitude")
            biz_lon = biz.get("coordinates", {}).get("longitude")
            if biz_lat is None or biz_lon is None:
                continue
            biz_aliases = [c["alias"] for c in biz.get("categories", [])]
            matched_category = next(
                (cat for cat in CATEGORIES if cat in biz_aliases), "other"
            )
            all_records.append({
                "business_id":      biz.get("id"),
                "name":             biz.get("name"),
                "matched_category": matched_category,
                "yelp_categories":  ", ".join(biz_aliases),
                "rating":           biz.get("rating"),
                "review_count":     biz.get("review_count"),
                "lat":              biz_lat,
                "lon":              biz_lon,
            })
    except Exception as e:
        print(f"  Skipping {fname}: {e}")

print(f"  Loaded {len(all_records)} business records from cache.")

# ── DEDUPLICATE ───────────────────────────────────────────────────────────────
# The same business may appear in queries from multiple neighboring tract centroids

print("\nDeduplicating businesses...")
df = pd.DataFrame(all_records)
df = df.drop_duplicates(subset=["business_id"])

# ── SPATIAL JOIN: ASSIGN EACH BUSINESS TO ITS ACTUAL CENSUS TRACT ─────────────
# The queried_geoid above is the tract whose centroid we searched from.
# Here we assign each business to the tract it actually falls inside.

print("Spatially joining businesses to their actual census tracts...")
gdf_biz = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
    crs="EPSG:4326",
)

tracts_simple = tracts[["GEOID", "geometry"]].copy()
gdf_joined = gpd.sjoin(gdf_biz, tracts_simple, how="left", predicate="within")
gdf_joined = gdf_joined.rename(columns={"GEOID": "actual_geoid"})

# ── SAVE OUTPUT ───────────────────────────────────────────────────────────────
# Review date fetching skipped due to API rate limits.
# Use review_count > 5 as a quality filter instead when building features.

output_df = gdf_joined.drop(columns=["geometry", "index_right"], errors="ignore")
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone. {len(output_df)} records saved to {OUTPUT_CSV}")
print("Next step: filter by review_count > 5 and compute")
print("business density per square mile per census tract.")
