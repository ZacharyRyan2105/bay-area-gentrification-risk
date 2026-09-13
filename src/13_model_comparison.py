"""
model_comparison.py
-------------------
Compares LASSO, XGBoost, and Random Forest across three datasets:
  - named_cities  (2015 vintage only)
  - 1-year appreciation model
  - 5-year appreciation model

All three datasets use the same binary gentrification label:
  gentrified = 1  iff  below_city_median_income == 1
                  AND  appreciation is above the within-city 75th percentile
                       (threshold computed on training data only)

  - named_cities: appreciation = (jan_housing_value_end - jan_housing_value_begin)
                                  / jan_housing_value_begin
  - 1yr:          appreciation = appreciation_1yr
  - 5yr:          appreciation = appreciation_5yr

Named_cities is restricted to 2015 rows to avoid 2008 recession data.
All datasets are restricted to below_city_median_income == 1 tracts.
No census_multivintage data needed.

SMOTE is applied INSIDE each CV fold to prevent data leakage.
All models use the same BASE_FEATURES for a fair cross-dataset comparison.

Usage (terminal):
    python model_comparison.py

Usage (Jupyter / Colab):
    Set HERE manually before running, e.g.:
        HERE = Path("/content/drive/MyDrive/MS&E 125 Project")
    Then call main().

Outputs:
    model_comparison_results.csv   — full metrics table
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.base import clone
from sklearn.pipeline import Pipeline

import xgboost as xgb

# ── PATHS ──────────────────────────────────────────────────────────────────────
# In Jupyter/Colab, override HERE before calling main(), e.g.:
#   HERE = Path("/content/drive/MyDrive/MS&E 125 Project")
HERE = Path(__file__).parent

TRAIN_NC  = HERE / "training_named_cities.csv"
VAL_NC    = HERE / "validation_named_cities.csv"
TRAIN_1YR = HERE / "training_1yr.csv"
VAL_1YR   = HERE / "validation_1yr.csv"
TRAIN_5YR = HERE / "training_5yr.csv"
VAL_5YR   = HERE / "validation_5yr.csv"
OUT_CSV   = HERE / "model_comparison_results.csv"

# ── FEATURES ───────────────────────────────────────────────────────────────────
# All features are ACS census or LODES job data — no Zillow data.
# median_home_value is ACS (owner self-reported), NOT Zillow.
#
# Zillow columns exist in the raw files but are intentionally excluded:
#   named_cities : jan_housing_value_begin, jan_housing_value_end
#   1yr / 5yr    : jan_value_begin, appreciation_1yr, appreciation_5yr
# These are only used to compute the gentrification label, never as features.

ZILLOW_COLS = {
    "jan_housing_value_begin", "jan_housing_value_end",
    "jan_value_begin", "appreciation_1yr", "appreciation_5yr",
    "appreciation_nc",   # derived column added by this script
}

BASE_FEATURES = [
    # ACS census snapshot
    "median_income", "median_rent", "median_home_value",
    "share_college", "share_25_34", "homeownership_rate",
    "vacancy_rate", "share_pre1940", "share_white", "share_black",
    "share_hispanic", "poverty_rate", "rent_burden", "population",
    # LODES job density
    "arts_density", "food_density", "gentrify_density",
    "arts_job_share", "food_job_share",
    # City dummies
    "city_sf", "city_oakland", "city_san_jose",
]

# Safety check: BASE_FEATURES must never contain a Zillow column
assert not ZILLOW_COLS.intersection(BASE_FEATURES), \
    "Zillow column found in BASE_FEATURES — remove it before running."

# ── MODEL DEFINITIONS ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS     = 5

MODEL_SPECS = {
    "LASSO": LogisticRegression(
        penalty="l1", solver="liblinear", C=0.1,
        max_iter=1000, random_state=RANDOM_STATE,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1,
        class_weight="balanced"
    ),
    # XGBoost uses scale_pos_weight instead of class_weight.
    # Set dynamically in evaluate_model() based on training class ratio.
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0,
        random_state=RANDOM_STATE
    ),
}


def make_pipeline(model):
    """StandardScaler → model. Class imbalance handled via class_weight / scale_pos_weight."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clone(model)),
    ])


# ── DATA LOADING ───────────────────────────────────────────────────────────────

def load_all():
    print("Loading datasets...")
    dfs = {}
    for name, path in [
        ("train_nc",  TRAIN_NC),
        ("val_nc",    VAL_NC),
        ("train_1yr", TRAIN_1YR),
        ("val_1yr",   VAL_1YR),
        ("train_5yr", TRAIN_5YR),
        ("val_5yr",   VAL_5YR),
    ]:
        df = pd.read_csv(path)
        df["tract_id"] = df["tract_id"].astype(str).str.zfill(11)
        dfs[name] = df
        print(f"  {name:<12}: {len(df):,} rows")
    return dfs


def add_city_col(df):
    """Reconstruct a 'city' string column from one-hot dummies."""
    df = df.copy()
    df["city"] = "Other"
    if "city_sf"       in df.columns: df.loc[df["city_sf"]       == 1, "city"] = "San Francisco"
    if "city_oakland"  in df.columns: df.loc[df["city_oakland"]  == 1, "city"] = "Oakland"
    if "city_san_jose" in df.columns: df.loc[df["city_san_jose"] == 1, "city"] = "San Jose"
    return df


# ── GENTRIFICATION LABELING ────────────────────────────────────────────────────

def apply_gent_label(train_df, val_df, appr_col):
    """
    Assign binary gentrification label using appreciation only.

    A tract is gentrified if:
      1. below_city_median_income == 1
      2. appreciation (appr_col) is above the within-city 75th percentile
         for that dataset's own period.

    Thresholds are computed SEPARATELY for training and validation so that
    each period's label reflects relative performance among contemporaries,
    not against a historical threshold from a different market regime.
    This avoids the regime-shift problem where boom-era thresholds make it
    impossible for any validation tract to qualify in a cooler market.

    Each dataset will have approximately 25% positives among below-median
    tracts by construction, making cross-dataset comparisons meaningful.

    Returns (train_df, val_df) with a 'gentrified' column added.
    """
    def label_df(df):
        df = df.copy()
        city_thresh   = df.groupby("city")[appr_col].quantile(0.75)
        global_thresh = df[appr_col].quantile(0.75)
        thresh = df["city"].map(city_thresh).fillna(global_thresh)
        df["gentrified"] = (
            (df["below_city_median_income"] == 1) &
            (df[appr_col] > thresh)
        ).astype(int)
        return df

    return label_df(train_df), label_df(val_df)


# ── MODEL EVALUATION ───────────────────────────────────────────────────────────

def evaluate_model(train_df, val_df, model, model_name, dataset_name):
    """Train and evaluate one model on one dataset. Returns a metrics dict."""
    feats = [f for f in BASE_FEATURES
             if f in train_df.columns and f in val_df.columns
             and f not in ZILLOW_COLS]   # belt-and-suspenders: never predict with Zillow data

    # Fill missing values with training medians
    train_medians = train_df[feats].median()
    X_train = train_df[feats].fillna(train_medians).values
    y_train = train_df["gentrified"].values
    X_val   = val_df[feats].fillna(train_medians).values
    y_val   = val_df["gentrified"].values

    # Guard: need at least 2 classes in both splits
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print(f"    [{dataset_name} / {model_name}] Skipped — only one class present.")
        return None

    # XGBoost: set scale_pos_weight = n_negative / n_positive (equivalent to balanced)
    if model_name == "XGBoost":
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        model = clone(model)
        model.set_params(scale_pos_weight=n_neg / n_pos)

    pipe = make_pipeline(model)

    # Cross-validation (SMOTE applied inside each fold — no leakage)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_aucs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    # Fit on full training set, evaluate on held-out validation
    pipe.fit(X_train, y_train)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    val_pred  = pipe.predict(X_val)

    val_auc = roc_auc_score(y_val, val_proba)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, val_pred, zero_division=0
    )

    return {
        "Dataset":         dataset_name,
        "Model":           model_name,
        "Train +":         int(y_train.sum()),
        "Train total":     len(y_train),
        "Val +":           int(y_val.sum()),
        "Val total":       len(y_val),
        "CV AUC (mean)":   round(cv_aucs.mean(), 4),
        "CV AUC (std)":    round(cv_aucs.std(),  4),
        "Val AUC":         round(val_auc, 4),
        "Val Precision-1": round(prec[1], 3) if len(prec) > 1 else 0,
        "Val Recall-1":    round(rec[1],  3) if len(rec)  > 1 else 0,
        "Val F1-1":        round(f1[1],   3) if len(f1)   > 1 else 0,
    }


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    dfs = load_all()

    # ── named_cities: 2015 rows only, below-median only ───────────────────
    print("\nPreparing named_cities (2015 vintage, below-median tracts)...")
    train_nc = dfs["train_nc"][
        (dfs["train_nc"]["year"] == 2015) &
        (dfs["train_nc"]["below_city_median_income"] == 1)
    ].copy()
    val_nc = dfs["val_nc"][
        dfs["val_nc"]["below_city_median_income"] == 1
    ].copy()

    train_nc = add_city_col(train_nc)
    val_nc   = add_city_col(val_nc)

    # Compute appreciation from Zillow begin/end columns
    for df in (train_nc, val_nc):
        df["appreciation_nc"] = (
            (df["jan_housing_value_end"] - df["jan_housing_value_begin"])
            / df["jan_housing_value_begin"]
        )

    train_nc, val_nc = apply_gent_label(train_nc, val_nc, "appreciation_nc")

    # ── 1yr: drop 2013 and 2014 vintages (ACS 5-yr estimates for those years
    #    include 2008-2009 recession data in their rolling averages) ─────────
    print("Preparing 1yr datasets (vintages 2015+ only)...")
    train_1yr = add_city_col(
        dfs["train_1yr"][dfs["train_1yr"]["vintage_year"] >= 2015].copy()
    )
    val_1yr   = add_city_col(dfs["val_1yr"])
    train_1yr, val_1yr = apply_gent_label(train_1yr, val_1yr, "appreciation_1yr")

    # ── 5yr: below-median already filtered, add city col, label ───────────
    print("Preparing 5yr datasets...")
    train_5yr = add_city_col(dfs["train_5yr"])
    val_5yr   = add_city_col(dfs["val_5yr"])
    train_5yr, val_5yr = apply_gent_label(train_5yr, val_5yr, "appreciation_5yr")

    # ── Label distribution summary ─────────────────────────────────────────
    print("\nLabel distributions:")
    for name, tr, va in [
        ("named_cities", train_nc,  val_nc),
        ("1yr",          train_1yr, val_1yr),
        ("5yr",          train_5yr, val_5yr),
    ]:
        tp, tt = int(tr["gentrified"].sum()), len(tr)
        vp, vt = int(va["gentrified"].sum()), len(va)
        print(f"  {name:<14}  train: {tp:>3}/{tt}  ({100*tp/tt:.1f}%)   "
              f"val: {vp:>3}/{vt}  ({100*vp/vt:.1f}%)")

    # ── Run all 9 combinations ─────────────────────────────────────────────
    datasets = {
        "named_cities": (train_nc,  val_nc),
        "1yr":          (train_1yr, val_1yr),
        "5yr":          (train_5yr, val_5yr),
    }

    print("\nRunning models...")
    results = []
    for ds_name, (tr, va) in datasets.items():
        for model_name, model in MODEL_SPECS.items():
            print(f"  {ds_name:<14} × {model_name}...")
            result = evaluate_model(tr, va, model, model_name, ds_name)
            if result:
                results.append(result)

    # ── Results table ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("MODEL COMPARISON RESULTS")
    print("=" * 90)
    print(results_df.to_string(index=False))

    results_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")

    # ── Summary pivots ─────────────────────────────────────────────────────
    print("\n── Val AUC ──")
    print(results_df.pivot(index="Dataset", columns="Model", values="Val AUC").to_string())

    print("\n── Val Recall (class 1) ──")
    print(results_df.pivot(index="Dataset", columns="Model", values="Val Recall-1").to_string())

    print("\n── Val F1 (class 1) ──")
    print(results_df.pivot(index="Dataset", columns="Model", values="Val F1-1").to_string())


if __name__ == "__main__":
    main()
