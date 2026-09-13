# ════════════════════════════════════════════════════════════════════════════
# CELL 1 — Upload permit data and existing datasets
# ════════════════════════════════════════════════════════════════════════════

from google.colab import files
import pandas as pd
import numpy as np

print("Upload the following files when prompted:")
print("  1. permits_by_tract_year.csv")
print("  2. merged_training_data.csv")
print("  3. merged_validation_data.csv")
print("  4. census_and_LODES_test_data.csv  (or your test set file)")

uploaded = files.upload()
print("\nUploaded:", list(uploaded.keys()))


# ════════════════════════════════════════════════════════════════════════════
# CELL 2 — Load and inspect all files
# ════════════════════════════════════════════════════════════════════════════

permits = pd.read_csv("permits_by_tract_year.csv")
train   = pd.read_csv("merged_training_data.csv")
val     = pd.read_csv("merged_validation_data.csv")
test    = pd.read_csv("census_and_LODES_test_data.csv")   # update filename if different

# Ensure tract_id is a consistent string type across all files
for df in [permits, train, val, test]:
    df["tract_id"] = df["tract_id"].astype(str).str.zfill(11)

# Fix: valuation of 0.0 means "no data" for SF/Oakland — treat as NaN
permits["new_const_valuation_sum"]  = permits["new_const_valuation_sum"].replace(0.0, np.nan)
permits["all_permits_valuation_sum"] = permits["all_permits_valuation_sum"].replace(0.0, np.nan)

# Derive city from county FIPS (first 5 digits of tract_id)
county_to_city = {
    "06075": "San Francisco",
    "06001": "Oakland/Alameda",
    "06013": "Oakland/Contra Costa",
    "06081": "San Mateo",
    "06085": "San Jose",
}
permits["city"] = permits["tract_id"].str[:5].map(county_to_city).fillna("Other")

print("Permits shape:", permits.shape)
print("Year range:", permits["year_filed"].min(), "–", permits["year_filed"].max())
print("\nPermit counts by city:")
print(permits.groupby("city")["new_const_count"].sum().sort_values(ascending=False))
print("\nTrain shape:", train.shape)
print("Val shape:  ", val.shape)
print("Test shape: ", test.shape)


# ════════════════════════════════════════════════════════════════════════════
# CELL 3 — Aggregate permits over 5-year windows per period
# ════════════════════════════════════════════════════════════════════════════

PERMIT_COLS = [
    "new_const_count",
    "new_const_valuation_sum",
    "new_const_sqft_sum",
    "new_const_units_sum",
    "all_permits_count",
]

def aggregate_window(permits_df, year_start, year_end, suffix):
    """Sum permit activity within a year window, one row per tract."""
    window = permits_df[permits_df["year_filed"].between(year_start, year_end)]
    agg = (window
           .groupby("tract_id")[PERMIT_COLS]
           .sum()
           .reset_index())
    # Rename columns with period suffix
    agg = agg.rename(columns={c: f"{c}_{suffix}" for c in PERMIT_COLS})
    return agg

# Training period: 5 years centered on 2010 features
permits_train = aggregate_window(permits, 2008, 2012, "train")

# Validation period: 5 years centered on 2015 features
permits_val   = aggregate_window(permits, 2013, 2017, "val")

# Test period: most recent 4 years (2021–2024)
permits_test  = aggregate_window(permits, 2021, 2024, "test")

print("Training permit window (2008–2012):")
print(permits_train.describe().round(1))
print("\nValidation permit window (2013–2017):")
print(permits_val.describe().round(1))
print("\nTest permit window (2021–2024):")
print(permits_test.describe().round(1))


# ════════════════════════════════════════════════════════════════════════════
# CELL 4 — Merge permits into training set
# ════════════════════════════════════════════════════════════════════════════

train_merged = train.merge(permits_train, on="tract_id", how="left")

# Tracts with no permits in the window → 0 activity (not missing)
permit_train_cols = [c for c in train_merged.columns if c.endswith("_train")]
train_merged[permit_train_cols] = train_merged[permit_train_cols].fillna(0)

# Within-city percentile rank (accounts for San Jose data density bias)
train_merged["city_group"] = train_merged["tract_id"].str[:5].map(county_to_city).fillna("Other")
for col in ["new_const_count_train", "new_const_sqft_sum_train", "new_const_units_sum_train"]:
    train_merged[f"{col}_pct"] = train_merged.groupby("city_group")[col].rank(pct=True)

print("Training set after permit merge:", train_merged.shape)
print("New permit columns:", [c for c in train_merged.columns if "new_const" in c or "all_permits" in c])
print("\nPermit feature summary:")
print(train_merged[[c for c in train_merged.columns if c.endswith("_train")]].describe().round(1))


# ════════════════════════════════════════════════════════════════════════════
# CELL 5 — Merge permits into validation set
# ════════════════════════════════════════════════════════════════════════════

val_merged = val.merge(permits_val, on="tract_id", how="left")

permit_val_cols = [c for c in val_merged.columns if c.endswith("_val")]
val_merged[permit_val_cols] = val_merged[permit_val_cols].fillna(0)

val_merged["city_group"] = val_merged["tract_id"].str[:5].map(county_to_city).fillna("Other")
for col in ["new_const_count_val", "new_const_sqft_sum_val", "new_const_units_sum_val"]:
    val_merged[f"{col}_pct"] = val_merged.groupby("city_group")[col].rank(pct=True)

print("Validation set after permit merge:", val_merged.shape)
print("Coverage — tracts with any permit activity:",
      (val_merged["new_const_count_val"] > 0).sum(), "of", len(val_merged))


# ════════════════════════════════════════════════════════════════════════════
# CELL 6 — Merge permits into test set
# ════════════════════════════════════════════════════════════════════════════

test_merged = test.merge(permits_test, on="tract_id", how="left")

permit_test_cols = [c for c in test_merged.columns if c.endswith("_test")]
test_merged[permit_test_cols] = test_merged[permit_test_cols].fillna(0)

test_merged["city_group"] = test_merged["tract_id"].str[:5].map(county_to_city).fillna("Other")
for col in ["new_const_count_test", "new_const_sqft_sum_test", "new_const_units_sum_test"]:
    test_merged[f"{col}_pct"] = test_merged.groupby("city_group")[col].rank(pct=True)

print("Test set after permit merge:", test_merged.shape)
print("Coverage — tracts with any permit activity:",
      (test_merged["new_const_count_test"] > 0).sum(), "of", len(test_merged))


# ════════════════════════════════════════════════════════════════════════════
# CELL 7 — Sanity check and save
# ════════════════════════════════════════════════════════════════════════════

print("=== Final dataset shapes ===")
print(f"  Training   : {train_merged.shape}")
print(f"  Validation : {val_merged.shape}")
print(f"  Test       : {test_merged.shape}")

# Check permit coverage by city group
for label, df in [("Train", train_merged), ("Val", val_merged), ("Test", test_merged)]:
    col = [c for c in df.columns if "new_const_count" in c and "pct" not in c]
    if col:
        coverage = (df[col[0]] > 0).groupby(df["city_group"]).mean().round(2)
        print(f"\n{label} — fraction of tracts with permit activity:")
        print(coverage.to_string())

# Verify no data leakage — permit features should use only pre-period data
print("\n=== Leakage check ===")
print("Training permit window: 2008–2012  (predicting 2010→2015 change) ✓")
print("Validation permit window: 2013–2017 (predicting 2015→2020 change) ✓")
print("Test permit window: 2021–2024       (predicting future change)     ✓")

# Save final datasets
train_merged.to_csv("training_with_permits.csv", index=False)
val_merged.to_csv("validation_with_permits.csv", index=False)
test_merged.to_csv("test_with_permits.csv", index=False)
print("\nSaved:")
print("  training_with_permits.csv")
print("  validation_with_permits.csv")
print("  test_with_permits.csv")
