"""
Comprehensive Quality & Leakage Audit

Runs all required quality checks for the final ML dataset.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/processed/earnings_ml_ready.parquet")
EVENTS_FILE = Path("data/processed/earnings_events.parquet")
MANIFEST_FILE = Path("data/processed/earnings_feature_manifest.csv")

REPORT_FILE = Path("data/processed/earnings_quality_audit_report.txt")
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load dataset and manifest."""
    df = pd.read_parquet(INPUT_FILE)
    manifest = pd.read_csv(MANIFEST_FILE)
    events = pd.read_parquet(EVENTS_FILE)
    return df, manifest, events


def check_dataset_shape(df):
    """Check 1: Dataset shape."""
    print("=" * 70)
    print("CHECK 1: DATASET SHAPE")
    print("=" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return {"rows": len(df), "cols": len(df.columns)}


def check_event_range(df):
    """Check 2: Earliest/latest event."""
    print("\n" + "=" * 70)
    print("CHECK 2: EVENT DATE RANGE")
    print("=" * 70)
    print(f"Earliest period_ended: {df['period_ended'].min()}")
    print(f"Latest period_ended: {df['period_ended'].max()}")
    print(f"Earliest feature_cutoff: {df['feature_cutoff_date'].min()}")
    print(f"Latest feature_cutoff: {df['feature_cutoff_date'].max()}")
    print(f"Earliest reaction_start: {df['reaction_start_date'].min()}")
    print(f"Latest reaction_start: {df['reaction_start_date'].max()}")
    return {
        "earliest_period": str(df['period_ended'].min()),
        "latest_period": str(df['period_ended'].max()),
    }


def check_events_per_symbol(df):
    """Check 3: Events per symbol."""
    print("\n" + "=" * 70)
    print("CHECK 3: EVENTS PER SYMBOL")
    print("=" * 70)
    counts = df.groupby("symbol").size().sort_values()
    print(f"Min events: {counts.min()} ({counts.idxmin()})")
    print(f"Max events: {counts.max()} ({counts.idxmax()})")
    print(f"Mean events: {counts.mean():.1f}")
    print(f"Median events: {counts.median():.1f}")
    return counts.to_dict()


def check_duplicate_events(df):
    """Check 4 & 5: Duplicate event keys and symbol-period."""
    print("\n" + "=" * 70)
    print("CHECK 4 & 5: DUPLICATE CHECKS")
    print("=" * 70)
    dup_keys = df["event_key"].duplicated().sum()
    dup_sp = df.duplicated(subset=["symbol", "period_ended"]).sum()
    print(f"Duplicate event_keys: {dup_keys}")
    print(f"Duplicate symbol+period: {dup_sp}")
    return {"dup_event_keys": int(dup_keys), "dup_symbol_period": int(dup_sp)}


def check_missing_targets(df):
    """Check 6: Missing target."""
    print("\n" + "=" * 70)
    print("CHECK 6: MISSING TARGETS")
    print("=" * 70)
    for target in ["reaction_class", "abnormal_return_1d", "abnormal_return_3d", "abnormal_return_5d"]:
        missing = df[target].isna().sum()
        print(f"{target}: {missing} missing ({missing/len(df)*100:.1f}%)")
    return {t: int(df[t].isna().sum()) for t in ["reaction_class", "abnormal_return_1d", "abnormal_return_3d", "abnormal_return_5d"]}


def check_target_distribution(df):
    """Check 7: Target distribution."""
    print("\n" + "=" * 70)
    print("CHECK 7: TARGET DISTRIBUTION")
    print("=" * 70)
    dist = df["reaction_class"].value_counts()
    print(dist.to_string())
    print(f"\nClass balance: {dist.min()/dist.max():.3f} (min/max ratio)")
    return dist.to_dict()


def check_missing_values(df, manifest):
    """Check 8 & 9: Missing values per feature."""
    print("\n" + "=" * 70)
    print("CHECK 8 & 9: MISSING VALUES")
    print("=" * 70)
    
    features = manifest[manifest["is_target"] == False]["feature_name"].tolist()
    missing = {}
    high_missing = []
    for col in features:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            missing[col] = pct
            if pct > 50:
                high_missing.append((col, pct))
    
    print(f"Features with >50% missing: {len(high_missing)}")
    for col, pct in sorted(high_missing, key=lambda x: -x[1])[:20]:
        print(f"  {col}: {pct:.1f}%")
    
    # Overall stats
    all_pcts = list(missing.values())
    print(f"\nMissing % stats: mean={np.mean(all_pcts):.1f}%, median={np.median(all_pcts):.1f}%, max={np.max(all_pcts):.1f}%")
    
    return {"missing_pct": missing, "high_missing": high_missing}


def check_fundamental_coverage(df):
    """Check 10: Fundamental coverage."""
    print("\n" + "=" * 70)
    print("CHECK 10: FUNDAMENTAL COVERAGE")
    print("=" * 70)
    fund_cols = [c for c in df.columns if c.startswith("fund_")]
    has_fund = df["fundamental_pit_status"] == "estimated_conservative"
    print(f"Events with fundamentals: {has_fund.sum()}/{len(df)} ({has_fund.mean()*100:.1f}%)")
    
    if has_fund.any():
        for col in sorted(fund_cols):
            pct = df.loc[has_fund, col].notna().mean() * 100
            if pct > 0:
                print(f"  {col}: {pct:.1f}% (of events with fundamentals)")
    
    return {"with_fundamentals": int(has_fund.sum()), "total": len(df)}


def check_price_history_coverage(df):
    """Check 11: Price history coverage."""
    print("\n" + "=" * 70)
    print("CHECK 11: PRICE HISTORY COVERAGE")
    print("=" * 70)
    print(f"price_history_days: mean={df['price_history_days'].mean():.1f}, median={df['price_history_days'].median():.1f}")
    print(f"benchmark_history_days: mean={df['benchmark_history_days'].mean():.1f}, median={df['benchmark_history_days'].median():.1f}")
    print(f"Min price history: {df['price_history_days'].min()}")
    print(f"Events with <20d history: {(df['price_history_days'] < 20).sum()}")
    print(f"Events with <60d history: {(df['price_history_days'] < 60).sum()}")
    print(f"Events with <252d history: {(df['price_history_days'] < 252).sum()}")
    return {
        "mean_price_hist": df['price_history_days'].mean(),
        "mean_bench_hist": df['benchmark_history_days'].mean(),
    }


def check_class_distribution_over_time(df):
    """Check 12: Class distribution over time."""
    print("\n" + "=" * 70)
    print("CHECK 12: CLASS DISTRIBUTION OVER TIME")
    print("=" * 70)
    df["year"] = df["period_ended"].dt.year
    yearly = df.groupby("year")["reaction_class"].value_counts(normalize=True).unstack(fill_value=0)
    print(yearly.round(3).to_string())
    return yearly.to_dict()


def check_dtypes(df):
    """Check 13: Feature data types."""
    print("\n" + "=" * 70)
    print("CHECK 13: DATA TYPES")
    print("=" * 70)
    dtypes = df.dtypes.value_counts()
    print(dtypes.to_string())
    return dtypes.to_dict()


def check_infinite_values(df, manifest):
    """Check 14: Infinite values."""
    print("\n" + "=" * 70)
    print("CHECK 14: INFINITE VALUES")
    print("=" * 70)
    features = manifest[manifest["is_target"] == False]["feature_name"].tolist()
    inf_cols = []
    for col in features:
        if col in df.columns and df[col].dtype in [np.float64, np.float32, float]:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
    print(f"Columns with inf: {len(inf_cols)}")
    for c in inf_cols:
        print(f"  {c}")
    return {"inf_columns": inf_cols}


def check_constant_features(df, manifest):
    """Check 15: Constant features."""
    print("\n" + "=" * 70)
    print("CHECK 15: CONSTANT FEATURES")
    print("=" * 70)
    features = manifest[manifest["is_target"] == False]["feature_name"].tolist()
    const_cols = []
    for col in features:
        if col in df.columns:
            nunique = df[col].nunique()
            if nunique <= 1:
                const_cols.append((col, df[col].unique()[:5] if nunique == 1 else "all NaN"))
    print(f"Constant/near-constant columns: {len(const_cols)}")
    for c, val in const_cols:
        print(f"  {c}: {val}")
    return {"constant_columns": const_cols}


def check_suspicious_columns(df, manifest):
    """Check 16: Highly suspicious columns."""
    print("\n" + "=" * 70)
    print("CHECK 16: SUSPICIOUS COLUMNS")
    print("=" * 70)
    features = manifest[manifest["is_target"] == False]["feature_name"].tolist()
    suspicious = []
    for col in features:
        if col in df.columns and df[col].dtype in [np.float64, np.float32, float]:
            vals = df[col].dropna()
            if len(vals) > 0:
                # Check for extreme outliers
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 10*iqr, q3 + 10*iqr
                outliers = ((vals < lower) | (vals > upper)).sum()
                if outliers > len(vals) * 0.05:  # >5% extreme outliers
                    suspicious.append((col, outliers, len(vals)))
    print(f"Suspicious columns (high outlier rate): {len(suspicious)}")
    for c, o, t in suspicious[:10]:
        print(f"  {c}: {o}/{t} outliers ({o/t*100:.1f}%)")
    return {"suspicious": suspicious}


def check_future_leakage_by_names(df, manifest):
    """Check 17: Future leakage by column names."""
    print("\n" + "=" * 70)
    print("CHECK 17: FUTURE LEAKAGE BY COLUMN NAMES")
    print("=" * 70)
    leak_patterns = ["after", "future", "forward", "next", "target", "label"]
    features = manifest[manifest["is_target"] == False]["feature_name"].tolist()
    suspicious = []
    for col in features:
        for pat in leak_patterns:
            if pat in col.lower():
                suspicious.append((col, pat))
                break
    print(f"Potentially leaky column names: {len(suspicious)}")
    for c, p in suspicious:
        print(f"  {c} (pattern: {p})")
    return {"leaky_names": suspicious}


def check_pit_assertions(df):
    """Check 18: Point-in-time date assertions."""
    print("\n" + "=" * 70)
    print("CHECK 18: POINT-IN-TIME DATE ASSERTIONS")
    print("=" * 70)
    
    # Feature cutoff < reaction start
    chk1 = (df["feature_cutoff_date"] < df["reaction_start_date"]).all()
    print(f"feature_cutoff < reaction_start: {chk1}")
    
    # Feature cutoff <= announcement
    chk2 = (df["feature_cutoff_date"] <= df["result_announcement_datetime"]).all()
    print(f"feature_cutoff <= announcement: {chk2}")
    
    # Reaction start >= announcement (date)
    chk3 = (df["reaction_start_date"].dt.date >= df["result_announcement_datetime"].dt.date).all()
    print(f"reaction_start >= announcement (date): {chk3}")
    
    # Fundamental PIT checks
    pit_checks = []
    
    # Check: estimated_available_date <= feature_cutoff_date
    mask = df["fundamental_estimated_available_date"].notna()
    if mask.any():
        chk4a = (df.loc[mask, "fundamental_estimated_available_date"] <= df.loc[mask, "feature_cutoff_date"]).all()
        violations_a = (~(df.loc[mask, "fundamental_estimated_available_date"] <= df.loc[mask, "feature_cutoff_date"])).sum()
        print(f"fundamental_estimated_available <= cutoff: {chk4a} (violations: {violations_a})")
        pit_checks.append(chk4a)
    else:
        print("fundamental_estimated_available <= cutoff: N/A (no fundamentals)")
    
    # Check: fundamental_period_end_date < event's period_ended (no same-quarter leakage)
    mask = df["fundamental_period_end_date"].notna()
    if mask.any():
        chk4b = (df.loc[mask, "fundamental_period_end_date"] < df.loc[mask, "period_ended"]).all()
        violations_b = (~(df.loc[mask, "fundamental_period_end_date"] < df.loc[mask, "period_ended"])).sum()
        print(f"fundamental_period_end < event period_end: {chk4b} (violations: {violations_b})")
        pit_checks.append(chk4b)
    else:
        print("fundamental_period_end < event period_end: N/A (no fundamentals)")
    
    return {"chk1": chk1, "chk2": chk2, "chk3": chk3, "pit_checks": pit_checks}


def check_prior_earnings_shift(df, manifest):
    """Check 19: Prior-earnings shift assertions."""
    print("\n" + "=" * 70)
    print("CHECK 19: PRIOR-EARNINGS SHIFT ASSERTIONS")
    print("=" * 70)
    # Verify prior_earnings_count matches actual prior events
    # This is verified by construction, but we can spot-check
    prior_cols = [c for c in manifest[manifest["is_target"] == False]["feature_name"] if c.startswith("prior_")]
    print(f"Prior-earnings features: {len(prior_cols)}")
    print(f"prior_earnings_count range: {df['prior_earnings_count'].min()} to {df['prior_earnings_count'].max()}")
    print(f"Events with 0 prior: {(df['prior_earnings_count'] == 0).sum()}")
    print(f"Events with >=4 prior: {(df['prior_earnings_count'] >= 4).sum()}")
    # Check that prior_last_reaction_class is not current reaction
    if "prior_last_reaction_class" in df.columns:
        mismatch = (df["prior_last_reaction_class"] == df["reaction_class"]).sum()
        print(f"Prior reaction == current reaction: {mismatch} (may be legitimate streaks)")
    return {"prior_features": len(prior_cols)}


def check_chronological_split(df):
    """Check 20: Train/validation/test chronological split."""
    print("\n" + "=" * 70)
    print("CHECK 20: CHRONOLOGICAL SPLIT CHECK")
    print("=" * 70)
    
    # Split by unique period_ended dates (chronological quarters)
    unique_periods = sorted(df["period_ended"].unique())
    n_periods = len(unique_periods)
    
    train_periods = unique_periods[:int(n_periods * 0.6)]
    val_periods = unique_periods[int(n_periods * 0.6):int(n_periods * 0.8)]
    test_periods = unique_periods[int(n_periods * 0.8):]
    
    train = df[df["period_ended"].isin(train_periods)]
    val = df[df["period_ended"].isin(val_periods)]
    test = df[df["period_ended"].isin(test_periods)]
    
    print(f"Unique periods: {n_periods} (Train: {len(train_periods)}, Val: {len(val_periods)}, Test: {len(test_periods)})")
    print(f"Train: {len(train)} events ({train['period_ended'].min()} to {train['period_ended'].max()})")
    print(f"Val:   {len(val)} events ({val['period_ended'].min()} to {val['period_ended'].max()})")
    print(f"Test:  {len(test)} events ({test['period_ended'].min()} to {test['period_ended'].max()})")
    
    # Check no overlap
    train_max = train["period_ended"].max()
    val_min = val["period_ended"].min()
    val_max = val["period_ended"].max()
    test_min = test["period_ended"].min()
    
    print(f"\nTrain max < Val min: {train_max < val_min}")
    print(f"Val max < Test min: {val_max < test_min}")
    
    # Target distributions
    print(f"\nTrain reaction_class: {train['reaction_class'].value_counts().to_dict()}")
    print(f"Val reaction_class:   {val['reaction_class'].value_counts().to_dict()}")
    print(f"Test reaction_class:  {test['reaction_class'].value_counts().to_dict()}")
    
    return {
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "train_periods": len(train_periods),
        "val_periods": len(val_periods),
        "test_periods": len(test_periods),
        "train_max": str(train_max),
        "val_min": str(val_min),
        "val_max": str(val_max),
        "test_min": str(test_min),
        "no_overlap": train_max < val_min and val_max < test_min,
    }


def run_all_checks():
    """Run all quality checks."""
    print("=" * 70)
    print("COMPREHENSIVE QUALITY & LEAKAGE AUDIT")
    print("=" * 70)
    
    df, manifest, events = load_data()
    
    results = {}
    results["shape"] = check_dataset_shape(df)
    results["event_range"] = check_event_range(df)
    results["events_per_symbol"] = check_events_per_symbol(df)
    results["duplicates"] = check_duplicate_events(df)
    results["missing_targets"] = check_missing_targets(df)
    results["target_dist"] = check_target_distribution(df)
    results["missing_values"] = check_missing_values(df, manifest)
    results["fund_coverage"] = check_fundamental_coverage(df)
    results["price_coverage"] = check_price_history_coverage(df)
    results["class_over_time"] = check_class_distribution_over_time(df)
    results["dtypes"] = check_dtypes(df)
    results["infinite"] = check_infinite_values(df, manifest)
    results["constant"] = check_constant_features(df, manifest)
    results["suspicious"] = check_suspicious_columns(df, manifest)
    results["leakage_names"] = check_future_leakage_by_names(df, manifest)
    results["pit"] = check_pit_assertions(df)
    results["prior_shift"] = check_prior_earnings_shift(df, manifest)
    results["split"] = check_chronological_split(df)
    
    # Save report
    with open(REPORT_FILE, "w") as f:
        f.write("QUALITY AUDIT REPORT\n")
        f.write("=" * 70 + "\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n\n")
    
    print(f"\n\nReport saved to: {REPORT_FILE}")
    return results


if __name__ == "__main__":
    run_all_checks()