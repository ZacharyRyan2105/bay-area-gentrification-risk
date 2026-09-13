"""
get_tract_areas.py
------------------
Downloads the 2010 Census TIGER tract shapefile for California,
extracts land area (ALAND10) for all Bay Area tracts, and saves
tract_areas.csv — upload this file to Google Colab.

Requirements: requests (no geopandas needed)
"""

import requests, zipfile, io, struct, os
import pandas as pd

BAY_AREA = {"06001", "06013", "06075", "06081", "06085"}
OUT_CSV  = "tract_areas.csv"

# ── DOWNLOAD 2010 TIGER TRACT SHAPEFILE FOR CALIFORNIA ────────────────────────

url = "https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_06_tract10.zip"
print("Downloading 2010 TIGER tract shapefile for California...")
resp = requests.get(url, timeout=120)
print(f"  Status: {resp.status_code}  Size: {len(resp.content)/1e6:.1f} MB")

zf = zipfile.ZipFile(io.BytesIO(resp.content))
dbf_name = [n for n in zf.namelist() if n.endswith(".dbf")][0]
print(f"  Reading: {dbf_name}")
dbf_bytes = zf.read(dbf_name)

# ── PARSE DBF WITHOUT GEOPANDAS ───────────────────────────────────────────────

with io.BytesIO(dbf_bytes) as f:
    header = f.read(32)
    num_records = struct.unpack("<I", header[4:8])[0]
    header_size = struct.unpack("<H", header[8:10])[0]
    record_size = struct.unpack("<H", header[10:12])[0]

    fields = []
    while True:
        fd = f.read(32)
        if fd[0] == 0x0D:
            break
        name = fd[:11].replace(b"\x00", b"").decode("ascii")
        flen = fd[16]
        fields.append((name, flen))

    f.seek(header_size)
    records = []
    for _ in range(num_records):
        raw = f.read(record_size)
        row = {}
        pos = 1
        for name, flen in fields:
            row[name] = raw[pos:pos + flen].decode("latin-1").strip()
            pos += flen
        records.append(row)

df = pd.DataFrame(records)

# ── EXTRACT GEOID AND AREA ─────────────────────────────────────────────────────

# 2010 TIGER uses GEOID10 and ALAND10
geoid_col = "GEOID10" if "GEOID10" in df.columns else "CTIDFP10"
aland_col = "ALAND10" if "ALAND10" in df.columns else "ALAND"

df["tract_id"]  = df[geoid_col].astype(str).str.zfill(11)
df["area_sqmi"] = pd.to_numeric(df[aland_col], errors="coerce") / 2_589_988

# Filter to Bay Area counties
df["county_fips"] = df["tract_id"].str[:5]
df = df[df["county_fips"].isin(BAY_AREA)][["tract_id", "area_sqmi"]].copy()

print(f"\nBay Area tracts extracted: {len(df):,}")
print(f"Sample:\n{df.head(5).to_string()}")

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV} — upload this file to Google Colab")
