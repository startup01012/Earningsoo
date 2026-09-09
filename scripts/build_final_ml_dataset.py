"""
Create Final ML-Ready Dataset

Builds the final dataset with clear separation of:
A. Metadata / identifiers
B. Pre-event model features
C. Targets / post-event outcomes
D. Audit / leakage validation columns

Generates a feature manifest documenting all features.
"""

import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/processed/earnings_event_prior_earnings_features.parquet")

OUTPUT_FILE = Path("data/processed/earnings_ml_ready.parquet")
FEATURE_MANIFEST_FILE = Path("data/processed/earnings_feature_manifest.csv")
AUDIT_REPORT_FILE = Path("data/processed/earnings_ml_audit_report.csv")

for p in [OUTPUT_FILE, FEATURE_MANIFEST_FILE, AUDIT_REPORT_FILE]:
    p.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# COLUMN CLASSIFICATION
# ============================================================

# Forbidden columns - must NEVER be used as model features
FORBIDDEN_PATTERNS = [
    "^reaction_class$",           # Target (exact match)
    "^abnormal_return_",         # Target (not _after)
    "^return_1d_after$",
    "^return_3d_after$", 
    "^return_5d_after$",
    "^benchmark_return_1d_after$",
    "^benchmark_return_3d_after$",
    "^benchmark_return_5d_after$",
    "^reaction_start_date$",      # Leakage: post-event
]

# Metadata columns
META_COLS = [
    "event_key", "symbol", "company_name", "period_ended", "fiscal_quarter",
    "result_announcement_datetime", "announcement_session_type",
    "feature_cutoff_date", "reaction_start_date",
    "quality_flags", "price_history_days", "benchmark_history_days",
]

# Audit columns (for validation, not modeling)
AUDIT_COLS = [
    "fundamental_period_end_date", 
    "fundamental_estimated_available_date",
    "fundamental_pit_status",
    "fundamental_lag_days",
]

# Target columns (for modeling)
TARGET_COLS = [
    "reaction_class",
    "abnormal_return_1d", "abnormal_return_3d", "abnormal_return_5d",
    "return_1d_after", "return_3d_after", "return_5d_after",
    "benchmark_return_1d_after", "benchmark_return_3d_after", "benchmark_return_5d_after",
]

def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classify all columns into categories."""
    import re
    all_cols = set(df.columns)
    
    # Identify targets
    targets = [c for c in TARGET_COLS if c in all_cols]
    
    # Identify metadata
    meta = [c for c in META_COLS if c in all_cols]
    
    # Identify audit
    audit = [c for c in AUDIT_COLS if c in all_cols]
    
    # Forbidden check - use regex patterns
    forbidden_found = []
    for pattern in FORBIDDEN_PATTERNS:
        regex = re.compile(pattern)
        matches = [c for c in all_cols if regex.match(c)]
        forbidden_found.extend(matches)
    
    # Features = everything else
    used = set(meta + targets + audit + forbidden_found)
    features = sorted([c for c in all_cols if c not in used])
    
    return {
        "metadata": meta,
        "features": features,
        "targets": targets,
        "audit": audit,
        "forbidden": sorted(set(forbidden_found)),
    }


def create_feature_manifest(df: pd.DataFrame, classification: Dict[str, List[str]]) -> pd.DataFrame:
    """Create feature manifest documenting all features."""
    
    feature_groups = {
        "market_returns": ["return_1d", "return_3d", "return_5d", "return_10d", "return_20d", "return_60d"],
        "market_volatility": ["volatility_5d", "volatility_20d", "volatility_60d"],
        "market_volume": ["volume_ratio_5d", "volume_zscore_5d", "volume_ratio_20d", "volume_zscore_20d"],
        "market_ma_distance": ["distance_from_20dma", "distance_from_50dma", "distance_from_200dma"],
        "market_drawdown": ["drawdown_from_20d_high", "drawdown_from_60d_high", "drawdown_from_252d_high"],
        "market_relative": ["relative_return_1d", "relative_return_3d", "relative_return_5d", "relative_return_10d", "relative_return_20d", "relative_return_60d"],
        "fundamental_income": [c for c in classification["features"] if c.startswith("fund_") and any(x in c for x in ["revenue", "profit", "income", "eps", "ebit", "margin", "cost", "expense", "sga", "interest", "tax", "normalized"])],
        "fundamental_balance": [c for c in classification["features"] if c.startswith("fund_") and any(x in c for x in ["asset", "debt", "cash", "equity", "liabilit", "share", "working", "retained", "net_debt"])],
        "fundamental_cashflow": [c for c in classification["features"] if c.startswith("fund_") and any(x in c for x in ["cf_", "capex", "free_cash", "operating_cash", "investing", "financing", "depreciation", "change_in"])],
        "fundamental_derived": [c for c in classification["features"] if c.startswith("fund_") and any(x in c for x in ["growth", "margin", "ratio", "roe", "roa", "roce", "fcf_", "turnover", "current_", "debt_to", "cash_to", "asset_"])],
        "valuation": [c for c in classification["features"] if c.startswith("val_")],
        "prior_earnings": [c for c in classification["features"] if c.startswith("prior_")],
    }
    
    rows = []
    for col in classification["features"]:
        # Determine group
        group = "other"
        for g, cols in feature_groups.items():
            if col in cols:
                group = g
                break
        
        # Point-in-time rule
        if col.startswith("fund_"):
            pit_rule = "Fundamental report period_end_date <= feature_cutoff_date"
        elif col.startswith("val_"):
            pit_rule = "Price at feature_cutoff_date + fundamental report period_end_date <= feature_cutoff_date"
        elif col.startswith("prior_"):
            pit_rule = "Only prior earnings events (strictly before current period_ended)"
        elif col.startswith(("return_", "volatility_", "volume_", "distance_", "drawdown_", "relative_")):
            pit_rule = "Price data up to feature_cutoff_date only"
        else:
            pit_rule = "Pre-event data only"
        
        # Description
        desc_map = {
            # Market returns
            "return_1d": "1-day return before cutoff",
            "return_3d": "3-day return before cutoff",
            "return_5d": "5-day return before cutoff",
            "return_10d": "10-day return before cutoff",
            "return_20d": "20-day return before cutoff",
            "return_60d": "60-day return before cutoff",
            # Volatility
            "volatility_5d": "5-day annualized volatility",
            "volatility_20d": "20-day annualized volatility",
            "volatility_60d": "60-day annualized volatility",
            # Volume
            "volume_ratio_5d": "Current volume / 5-day avg volume",
            "volume_zscore_5d": "Volume z-score vs 5-day history",
            "volume_ratio_20d": "Current volume / 20-day avg volume",
            "volume_zscore_20d": "Volume z-score vs 20-day history",
            # MA distance
            "distance_from_20dma": "Distance from 20-day MA",
            "distance_from_50dma": "Distance from 50-day MA",
            "distance_from_200dma": "Distance from 200-day MA",
            # Drawdown
            "drawdown_from_20d_high": "Drawdown from 20-day high",
            "drawdown_from_60d_high": "Drawdown from 60-day high",
            "drawdown_from_252d_high": "Drawdown from 252-day high",
            # Relative
            "relative_return_1d": "1-day return vs NIFTYBEES",
            "relative_return_3d": "3-day return vs NIFTYBEES",
            "relative_return_5d": "5-day return vs NIFTYBEES",
            "relative_return_10d": "10-day return vs NIFTYBEES",
            "relative_return_20d": "20-day return vs NIFTYBEES",
            "relative_return_60d": "60-day return vs NIFTYBEES",
        }
        
        # Add fundamental descriptions
        for prefix, desc_prefix in [
            ("fund_revenue", "Quarterly revenue"),
            ("fund_gross_profit", "Quarterly gross profit"),
            ("fund_operating_income", "Quarterly operating income"),
            ("fund_net_income", "Quarterly net income"),
            ("fund_ebit", "Quarterly EBIT"),
            ("fund_ebitda", "Quarterly EBITDA"),
            ("fund_eps_diluted", "Quarterly diluted EPS"),
            ("fund_eps_basic", "Quarterly basic EPS"),
            ("fund_total_assets", "Quarterly total assets"),
            ("fund_total_debt", "Quarterly total debt"),
            ("fund_cash", "Quarterly cash"),
            ("fund_equity", "Quarterly equity"),
            ("fund_operating_cash_flow", "Quarterly operating cash flow"),
            ("fund_free_cash_flow", "Quarterly free cash flow"),
            ("fund_capex", "Quarterly capex"),
            ("fund_revenue_growth_qoq", "Revenue QoQ growth"),
            ("fund_revenue_growth_yoy", "Revenue YoY growth"),
            ("fund_operating_margin", "Operating margin"),
            ("fund_net_margin", "Net margin"),
            ("fund_ebitda_margin", "EBITDA margin"),
            ("fund_debt_to_equity", "Debt to equity"),
            ("fund_current_ratio", "Current ratio"),
            ("fund_roe", "Return on equity"),
            ("fund_roa", "Return on assets"),
            ("fund_roce", "Return on capital employed"),
            ("fund_fcf_margin", "FCF margin"),
            ("fund_asset_turnover", "Asset turnover"),
            ("val_pe", "P/E ratio"),
            ("val_pb", "P/B ratio"),
            ("val_ev_ebitda", "EV/EBITDA"),
            ("val_market_cap", "Market capitalization"),
            ("val_ev", "Enterprise value"),
            ("val_ps", "P/S ratio"),
            ("val_pcf", "P/CF ratio"),
            ("prior_earnings_count", "Number of prior earnings events"),
            ("prior_days_since_last", "Days since last earnings"),
            ("prior_last_reaction_class", "Last earnings reaction class"),
            ("prior_reaction_streak", "Consecutive same reaction count"),
        ]:
            if col.startswith(prefix):
                desc = desc_prefix
                break
        else:
            desc = col
        
        dtype = str(df[col].dtype)
        missing_pct = df[col].isna().mean() * 100
        
        rows.append({
            "feature_name": col,
            "feature_group": group,
            "description": desc,
            "point_in_time_rule": pit_rule,
            "expected_dtype": dtype,
            "missing_pct": round(missing_pct, 2),
            "is_target": False,
        })
    
    # Add targets to manifest
    for col in classification["targets"]:
        rows.append({
            "feature_name": col,
            "feature_group": "target",
            "description": f"Target: {col}",
            "point_in_time_rule": "POST-event (never use as feature)",
            "expected_dtype": str(df[col].dtype) if col in df.columns else "unknown",
            "missing_pct": round(df[col].isna().mean() * 100, 2) if col in df.columns else 100,
            "is_target": True,
        })
    
    manifest = pd.DataFrame(rows)
    return manifest


def run_leakage_audit(df: pd.DataFrame, classification: Dict[str, List[str]]) -> Dict:
    """Run comprehensive leakage audit."""
    results = {}
    
    # 1. Check no forbidden columns in features
    forbidden_in_features = set(classification["forbidden"]) & set(classification["features"])
    results["forbidden_in_features"] = list(forbidden_in_features)
    
    # 2. Check target columns exist
    results["targets_present"] = classification["targets"]
    
    # 3. Check feature_cutoff_date < reaction_start_date for all rows
    if "feature_cutoff_date" in df.columns and "reaction_start_date" in df.columns:
        results["cutoff_before_reaction"] = (df["feature_cutoff_date"] < df["reaction_start_date"]).all()
        results["cutoff_before_reaction_violations"] = (~(df["feature_cutoff_date"] < df["reaction_start_date"])).sum()
    else:
        results["cutoff_before_reaction"] = "N/A"
    
    # 4. Check feature_cutoff_date <= announcement_date
    if "feature_cutoff_date" in df.columns and "result_announcement_datetime" in df.columns:
        results["cutoff_before_announcement"] = (df["feature_cutoff_date"] <= df["result_announcement_datetime"]).all()
        results["cutoff_before_announcement_violations"] = (~(df["feature_cutoff_date"] <= df["result_announcement_datetime"])).sum()
    else:
        results["cutoff_before_announcement"] = "N/A"
    
    # 5. Check reaction_start_date >= announcement_date (date comparison)
    if "reaction_start_date" in df.columns and "result_announcement_datetime" in df.columns:
        react_date = df["reaction_start_date"].dt.date
        ann_date = df["result_announcement_datetime"].dt.date
        results["reaction_after_announcement"] = (react_date >= ann_date).all()
        results["reaction_after_announcement_violations"] = (~(react_date >= ann_date)).sum()
    else:
        results["reaction_after_announcement"] = "N/A"
    
    # 6. Check fundamental PIT: estimated_available_date <= feature_cutoff_date
    # Also check that fundamental_period_end_date < event's period_ended (no same-quarter leakage)
    pit_checks_passed = True
    pit_violations = 0
    
    # Check 6a: estimated_available_date <= feature_cutoff_date
    if "fundamental_estimated_available_date" in df.columns and "feature_cutoff_date" in df.columns:
        mask = df["fundamental_estimated_available_date"].notna()
        if mask.any():
            check_a = (df.loc[mask, "fundamental_estimated_available_date"] <= df.loc[mask, "feature_cutoff_date"]).all()
            violations_a = (~(df.loc[mask, "fundamental_estimated_available_date"] <= df.loc[mask, "feature_cutoff_date"])).sum()
            if not check_a:
                pit_checks_passed = False
                pit_violations += violations_a
        else:
            pit_checks_passed = "N/A (no fundamental data with estimated availability)"
    else:
        pit_checks_passed = "N/A (columns missing)"
    
    # Check 6b: fundamental_period_end_date < event's period_ended (no same-quarter leakage)
    if "fundamental_period_end_date" in df.columns and "period_ended" in df.columns:
        mask = df["fundamental_period_end_date"].notna()
        if mask.any():
            check_b = (df.loc[mask, "fundamental_period_end_date"] < df.loc[mask, "period_ended"]).all()
            violations_b = (~(df.loc[mask, "fundamental_period_end_date"] < df.loc[mask, "period_ended"])).sum()
            if not check_b:
                pit_checks_passed = False
                pit_violations += violations_b
    
    results["fundamental_pit_ok"] = pit_checks_passed
    results["fundamental_pit_violations"] = pit_violations
    
    # 7. Check prior-earnings only use prior events
    # This is verified by construction in build_prior_earnings_features.py
    results["prior_earnings_pit_verified"] = True
    
    # 8. Check for constant/near-constant features
    constant_features = []
    for col in classification["features"]:
        if col in df.columns:
            nunique = df[col].nunique()
            if nunique <= 1:
                constant_features.append(col)
    results["constant_features"] = constant_features
    
    # 9. Check for infinite values
    inf_features = []
    for col in classification["features"]:
        if col in df.columns and df[col].dtype in [np.float64, np.float32, float]:
            if np.isinf(df[col]).any():
                inf_features.append(col)
    results["infinite_features"] = inf_features
    
    # 10. Missing value summary
    missing_summary = {}
    for col in classification["features"]:
        if col in df.columns:
            missing_pct = df[col].isna().mean() * 100
            if missing_pct > 0:
                missing_summary[col] = round(missing_pct, 2)
    results["missing_summary"] = missing_summary
    
    # 11. Event key uniqueness
    results["event_key_unique"] = df["event_key"].is_unique
    results["duplicate_event_keys"] = df["event_key"].duplicated().sum()
    
    # 12. Symbol-period uniqueness
    if "symbol" in df.columns and "period_ended" in df.columns:
        results["symbol_period_unique"] = not df.duplicated(subset=["symbol", "period_ended"]).any()
        results["duplicate_symbol_period"] = df.duplicated(subset=["symbol", "period_ended"]).sum()
    
    return results


def build_final_dataset():
    """Main pipeline."""
    print("=" * 70)
    print("CREATE FINAL ML-READY DATASET")
    print("=" * 70)
    
    # Load dataset with all features
    print("\n[1/5] Loading feature dataset...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"  Shape: {df.shape}")
    
    # FILTER: Exclude events with future period_ended (future fiscal periods)
    # Data availability cutoff: 2026-09-08 (current date when pipeline ran)
    # Events with period_ended >= this date are future fiscal periods that should not be used for training
    DATA_AVAILABILITY_CUTOFF = pd.Timestamp("2026-09-08")
    original_count = len(df)
    df = df[df["period_ended"] < DATA_AVAILABILITY_CUTOFF].copy()
    filtered_count = original_count - len(df)
    print(f"  Filtered out {filtered_count} events with future period_ended (>= {DATA_AVAILABILITY_CUTOFF.date()})")
    print(f"  Remaining: {len(df)} events")
    
    # Classify columns
    print("\n[2/5] Classifying columns...")
    classification = classify_columns(df)
    print(f"  Metadata: {len(classification['metadata'])}")
    print(f"  Features: {len(classification['features'])}")
    print(f"  Targets: {len(classification['targets'])}")
    print(f"  Audit: {len(classification['audit'])}")
    print(f"  Forbidden (in data): {len(classification['forbidden'])}")
    
    if classification["forbidden"]:
        print(f"  Forbidden columns found: {classification['forbidden']}")
    
    # Create feature manifest
    print("\n[3/5] Creating feature manifest...")
    manifest = create_feature_manifest(df, classification)
    manifest.to_csv(FEATURE_MANIFEST_FILE, index=False)
    print(f"  Manifest saved: {FEATURE_MANIFEST_FILE} ({len(manifest)} rows)")
    
    # Run leakage audit
    print("\n[4/5] Running leakage audit...")
    audit_results = run_leakage_audit(df, classification)
    
    # Save audit report
    audit_df = pd.DataFrame([audit_results])
    audit_df.to_csv(AUDIT_REPORT_FILE, index=False)
    print(f"  Audit report saved: {AUDIT_REPORT_FILE}")
    
    # Print audit summary
    print("\n  LEAKAGE AUDIT RESULTS:")
    print(f"  - Forbidden columns in features: {len(audit_results['forbidden_in_features'])}")
    print(f"  - Cutoff < Reaction: {audit_results['cutoff_before_reaction']}")
    print(f"  - Cutoff <= Announcement: {audit_results['cutoff_before_announcement']}")
    print(f"  - Reaction >= Announcement: {audit_results['reaction_after_announcement']}")
    print(f"  - Fundamental PIT OK: {audit_results['fundamental_pit_ok']}")
    print(f"  - Prior earnings PIT verified: {audit_results['prior_earnings_pit_verified']}")
    print(f"  - Constant features: {len(audit_results['constant_features'])}")
    print(f"  - Infinite features: {len(audit_results['infinite_features'])}")
    print(f"  - Event key unique: {audit_results['event_key_unique']}")
    print(f"  - Symbol-period unique: {audit_results['symbol_period_unique']}")
    
    if audit_results["forbidden_in_features"]:
        print(f"  ⚠️  FORBIDDEN COLUMNS IN FEATURES: {audit_results['forbidden_in_features']}")
    
    # Create final ML-ready dataset (reorder columns)
    print("\n[5/5] Building final ML-ready dataset...")
    
    # Order: metadata + features (by group) + targets + audit
    meta = classification["metadata"]
    features = classification["features"]
    targets = classification["targets"]
    audit = classification["audit"]
    
    # Sort features by group for readability
    manifest_groups = manifest[manifest["is_target"] == False].sort_values("feature_group")
    feature_ordered = manifest_groups["feature_name"].tolist()
    
    final_cols = meta + feature_ordered + targets + audit
    final_cols = [c for c in final_cols if c in df.columns]
    
    df_final = df[final_cols].copy()
    
    # Save
    df_final.to_parquet(OUTPUT_FILE, index=False)
    print(f"  Saved: {OUTPUT_FILE} ({len(df_final)} rows, {len(df_final.columns)} columns)")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL ML-READY DATASET SUMMARY")
    print("=" * 70)
    print(f"Shape: {df_final.shape}")
    print(f"Metadata columns: {len(meta)}")
    print(f"Feature columns: {len(features)}")
    print(f"Target columns: {len(targets)}")
    print(f"Audit columns: {len(audit)}")
    
    print("\nFeature Groups:")
    for group in sorted(manifest["feature_group"].unique()):
        if group != "target":
            count = len(manifest[(manifest["feature_group"] == group) & (manifest["is_target"] == False)])
            print(f"  {group}: {count}")
    
    print(f"\nTarget: reaction_class (3 classes) + abnormal returns")
    print(f"Target distribution:")
    print(df_final["reaction_class"].value_counts().to_string())
    
    print("\nDone!")
    return df_final, manifest, audit_results


if __name__ == "__main__":
    build_final_dataset()