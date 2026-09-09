"""
Build Point-in-Time Fundamental Features

For each earnings event, finds the latest fundamental data available
ON OR BEFORE the feature_cutoff_date and computes derived metrics.

CRITICAL PIT NOTE: yfinance provides fiscal period_end_date but NOT the actual
filing/publication date. We use a conservative minimum filing lag of 45 days
(per SEBI LODR: quarterly results within 45 days of quarter end) to estimate
the earliest possible public availability date.

Output: data/processed/earnings_event_fundamental_features.parquet
"""

import warnings
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

EVENTS_INPUT = Path("data/processed/earnings_event_market_features.parquet")
FUNDAMENTALS_DIR = Path("data/raw/fundamentals")
OUTPUT_FILE = Path("data/processed/earnings_event_fundamental_features.parquet")
QUALITY_REPORT = Path("data/processed/earnings_event_fundamental_features_quality_report.csv")

for p in [OUTPUT_FILE, QUALITY_REPORT]:
    p.parent.mkdir(parents=True, exist_ok=True)

# PIT CONFIGURATION
# Conservative minimum filing lag: SEBI LODR requires quarterly results within 45 days
# of quarter end. Actual filing may be later, but 45 days is the regulatory minimum.
MIN_FILING_LAG_DAYS = 45

# Column mapping: raw column name -> standardized name
# We only map columns that exist across most symbols
COLUMN_MAP = {
    # Income Statement (quarterly)
    "q_is_Total Revenue": "revenue",
    "q_is_Gross Profit": "gross_profit",
    "q_is_Operating Income": "operating_income",
    "q_is_Operating Expense": "operating_expense",
    "q_is_Net Income": "net_income",
    "q_is_Net Income Common Stockholders": "net_income_common",
    "q_is_EBIT": "ebit",
    "q_is_EBITDA": "ebitda",
    "q_is_Diluted EPS": "eps_diluted",
    "q_is_Basic EPS": "eps_basic",
    "q_is_Diluted Average Shares": "shares_diluted",
    "q_is_Basic Average Shares": "shares_basic",
    "q_is_Interest Expense": "interest_expense",
    "q_is_Tax Provision": "tax_provision",
    "q_is_Pretax Income": "pretax_income",
    "q_is_Cost Of Revenue": "cost_of_revenue",
    "q_is_Operating Revenue": "operating_revenue",
    "q_is_Selling General And Administration": "sga",
    "q_is_Interest Income": "interest_income",
    "q_is_Normalized Income": "normalized_income",
    "q_is_Normalized EBITDA": "normalized_ebitda",
    
    # Balance Sheet (quarterly)
    "q_bs_Total Assets": "total_assets",
    "q_bs_Total Debt": "total_debt",
    "q_bs_Cash And Cash Equivalents": "cash",
    "q_bs_Cash Cash Equivalents And Short Term Investments": "cash_and_sti",
    "q_bs_Stockholders Equity": "equity",
    "q_bs_Current Assets": "current_assets",
    "q_bs_Current Liabilities": "current_liabilities",
    "q_bs_Total Liabilities Net Minority Interest": "total_liabilities",
    "q_bs_Long Term Debt": "long_term_debt",
    "q_bs_Current Debt": "current_debt",
    "q_bs_Ordinary Shares Number": "shares_outstanding",
    "q_bs_Working Capital": "working_capital",
    "q_bs_Net Debt": "net_debt",
    "q_bs_Retained Earnings": "retained_earnings",
    "q_bs_Total Equity Gross Minority Interest": "total_equity",
    
    # Cash Flow (quarterly)
    "q_cf_Operating Cash Flow": "operating_cash_flow",
    "q_cf_Capital Expenditure": "capex",
    "q_cf_Free Cash Flow": "free_cash_flow",
    "q_cf_Investing Cash Flow": "investing_cash_flow",
    "q_cf_Financing Cash Flow": "financing_cash_flow",
    "q_cf_Change In Working Capital": "change_in_working_capital",
    "q_cf_Depreciation And Amortization": "depreciation_amortization",
    "q_cf_Net Income From Continuing Operations": "cf_net_income",
    
    # Annual (fallback)
    "a_is_Total Revenue": "a_revenue",
    "a_is_Gross Profit": "a_gross_profit",
    "a_is_Operating Income": "a_operating_income",
    "a_is_Net Income": "a_net_income",
    "a_is_Diluted EPS": "a_eps_diluted",
    "a_bs_Total Assets": "a_total_assets",
    "a_bs_Total Debt": "a_total_debt",
    "a_bs_Cash And Cash Equivalents": "a_cash",
    "a_bs_Stockholders Equity": "a_equity",
    "a_bs_Current Assets": "a_current_assets",
    "a_bs_Current Liabilities": "a_current_liabilities",
    "a_bs_Long Term Debt": "a_long_term_debt",
    "a_bs_Ordinary Shares Number": "a_shares_outstanding",
    "a_cf_Operating Cash Flow": "a_operating_cash_flow",
    "a_cf_Capital Expenditure": "a_capex",
    "a_cf_Free Cash Flow": "a_free_cash_flow",
}

# Derived metrics to compute
DERIVED_METRICS = [
    "revenue_growth_qoq",
    "revenue_growth_yoy",
    "operating_margin",
    "net_margin",
    "ebitda_margin",
    "eps_growth_qoq",
    "eps_growth_yoy",
    "net_income_growth_qoq",
    "net_income_growth_yoy",
    "debt_to_equity",
    "current_ratio",
    "cash_to_debt",
    "roe",
    "roa",
    "roce",
    "fcf_margin",
    "fcf_to_debt",
    "asset_turnover",
]

# ============================================================
# UTILITIES
# ============================================================

def load_all_fundamentals() -> pd.DataFrame:
    """Load and concatenate all fundamental files."""
    files = sorted(FUNDAMENTALS_DIR.glob("*.parquet"))
    all_dfs = []
    for f in files:
        symbol = f.stem
        df = pd.read_parquet(f)
        df["symbol"] = symbol
        # Ensure period_end_date is datetime
        df["period_end_date"] = pd.to_datetime(df["period_end_date"], errors="coerce")
        all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    fundamentals = pd.concat(all_dfs, ignore_index=True)
    fundamentals = fundamentals.dropna(subset=["period_end_date", "symbol"])
    fundamentals = fundamentals.sort_values(["symbol", "period_end_date"]).reset_index(drop=True)
    return fundamentals


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standardized names."""
    # Only rename columns that exist in the dataframe
    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    return df.rename(columns=rename_map)


def get_available_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Get mapping of standardized name -> actual column name in df."""
    reverse_map = {}
    for raw, std in COLUMN_MAP.items():
        if raw in df.columns:
            reverse_map[std] = raw
    return reverse_map


def safe_divide(num: pd.Series, den: pd.Series, fill: float = np.nan) -> pd.Series:
    """Safe division with zero/NaN handling."""
    result = num / den
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(fill)


def calculate_growth(current: pd.Series, previous: pd.Series) -> pd.Series:
    """Calculate growth rate: (current - previous) / abs(previous)."""
    # Handle zero/negative denominators
    with np.errstate(divide='ignore', invalid='ignore'):
        growth = (current - previous) / previous.abs()
    growth = growth.replace([np.inf, -np.inf], np.nan)
    return growth


def compute_derived_metrics(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Compute derived fundamental metrics for each symbol over time."""
    df = fundamentals.copy()
    df = df.sort_values(["symbol", "period_end_date"]).reset_index(drop=True)
    
    # Revenue growth QoQ
    if "revenue" in df.columns:
        df["revenue_prev_q"] = df.groupby("symbol")["revenue"].shift(1)
        df["revenue_growth_qoq"] = calculate_growth(df["revenue"], df["revenue_prev_q"])
    
    # Revenue growth YoY (4 quarters)
    if "revenue" in df.columns:
        df["revenue_prev_y"] = df.groupby("symbol")["revenue"].shift(4)
        df["revenue_growth_yoy"] = calculate_growth(df["revenue"], df["revenue_prev_y"])
    
    # Margins
    if "operating_income" in df.columns and "revenue" in df.columns:
        df["operating_margin"] = safe_divide(df["operating_income"], df["revenue"])
    if "net_income" in df.columns and "revenue" in df.columns:
        df["net_margin"] = safe_divide(df["net_income"], df["revenue"])
    if "ebitda" in df.columns and "revenue" in df.columns:
        df["ebitda_margin"] = safe_divide(df["ebitda"], df["revenue"])
    
    # EPS growth
    if "eps_diluted" in df.columns:
        df["eps_prev_q"] = df.groupby("symbol")["eps_diluted"].shift(1)
        df["eps_growth_qoq"] = calculate_growth(df["eps_diluted"], df["eps_prev_q"])
        df["eps_prev_y"] = df.groupby("symbol")["eps_diluted"].shift(4)
        df["eps_growth_yoy"] = calculate_growth(df["eps_diluted"], df["eps_prev_y"])
    
    # Net income growth
    if "net_income" in df.columns:
        df["net_income_prev_q"] = df.groupby("symbol")["net_income"].shift(1)
        df["net_income_growth_qoq"] = calculate_growth(df["net_income"], df["net_income_prev_q"])
        df["net_income_prev_y"] = df.groupby("symbol")["net_income"].shift(4)
        df["net_income_growth_yoy"] = calculate_growth(df["net_income"], df["net_income_prev_y"])
    
    # Balance sheet ratios
    if "total_debt" in df.columns and "equity" in df.columns:
        df["debt_to_equity"] = safe_divide(df["total_debt"], df["equity"])
    if "current_assets" in df.columns and "current_liabilities" in df.columns:
        df["current_ratio"] = safe_divide(df["current_assets"], df["current_liabilities"])
    if "cash" in df.columns and "total_debt" in df.columns:
        df["cash_to_debt"] = safe_divide(df["cash"], df["total_debt"])
    elif "cash_and_sti" in df.columns and "total_debt" in df.columns:
        df["cash_to_debt"] = safe_divide(df["cash_and_sti"], df["total_debt"])
    
    # Profitability ratios
    if "net_income" in df.columns and "equity" in df.columns:
        df["roe"] = safe_divide(df["net_income"], df["equity"])
    if "net_income" in df.columns and "total_assets" in df.columns:
        df["roa"] = safe_divide(df["net_income"], df["total_assets"])
    # ROCE = EBIT / (Total Assets - Current Liabilities)
    if "ebit" in df.columns and "total_assets" in df.columns and "current_liabilities" in df.columns:
        capital_employed = df["total_assets"] - df["current_liabilities"]
        df["roce"] = safe_divide(df["ebit"], capital_employed)
    
    # Cash flow metrics
    if "free_cash_flow" in df.columns and "revenue" in df.columns:
        df["fcf_margin"] = safe_divide(df["free_cash_flow"], df["revenue"])
    if "free_cash_flow" in df.columns and "total_debt" in df.columns:
        df["fcf_to_debt"] = safe_divide(df["free_cash_flow"], df["total_debt"])
    
    # Efficiency
    if "revenue" in df.columns and "total_assets" in df.columns:
        df["asset_turnover"] = safe_divide(df["revenue"], df["total_assets"])
    
    return df


def find_latest_fundamental_before_cutoff(
    symbol_fundamentals: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    event_period_end: pd.Timestamp = None
) -> Optional[pd.Series]:
    """
    Find the latest fundamental report available ON OR BEFORE cutoff_date.
    
    IMPORTANT: yfinance provides fiscal period_end_date but NOT the actual
    filing/publication date. We estimate the earliest possible public availability
    as period_end_date + MIN_FILING_LAG_DAYS (45 days per SEBI LODR).
    
    CRITICAL LEAKAGE PREVENTION:
    - Only fundamental reports with period_end_date < event_period_end are used
      (cannot use current quarter's financials to predict current quarter's earnings)
    - Only reports with estimated_available_date <= cutoff_date are used
      (filing must have happened before feature cutoff)
    
    Args:
        symbol_fundamentals: DataFrame with fundamental data for one symbol
        cutoff_date: feature_cutoff_date for the event
        event_period_end: period_ended for the earnings event (must not use same-quarter data)
    
    Returns:
        Latest eligible fundamental report or None
    """
    if symbol_fundamentals.empty:
        return None
    
    # Calculate estimated availability date for each fundamental report
    symbol_fundamentals = symbol_fundamentals.copy()
    symbol_fundamentals["estimated_available_date"] = (
        symbol_fundamentals["period_end_date"] + pd.Timedelta(days=MIN_FILING_LAG_DAYS)
    )
    
    # Filter: period_end_date must be STRICTLY BEFORE event's period_end
    # Cannot use current quarter's financials to predict current quarter's earnings
    if event_period_end is not None:
        symbol_fundamentals = symbol_fundamentals[
            symbol_fundamentals["period_end_date"] < event_period_end
        ]
    
    # Filter: estimated_available_date must be <= feature_cutoff_date
    available = symbol_fundamentals[
        symbol_fundamentals["estimated_available_date"] <= cutoff_date
    ]
    
    if available.empty:
        return None
    
    # Get the latest one (by period_end_date)
    latest = available.iloc[-1]
    return latest


def build_fundamental_features():
    """Main pipeline to build point-in-time fundamental features."""
    print("=" * 70)
    print("BUILD POINT-IN-TIME FUNDAMENTAL FEATURES")
    print("=" * 70)
    
    # Load ML events (with feature_cutoff_date)
    print("\n[1/5] Loading ML events...")
    events = pd.read_parquet(EVENTS_INPUT)
    print(f"  Events: {len(events)}")
    print(f"  Date range: {events['feature_cutoff_date'].min()} to {events['feature_cutoff_date'].max()}")
    
    # Load fundamentals
    print("\n[2/5] Loading fundamental data...")
    fundamentals = load_all_fundamentals()
    print(f"  Total rows: {len(fundamentals)}")
    print(f"  Symbols: {fundamentals['symbol'].nunique()}")
    print(f"  Date range: {fundamentals['period_end_date'].min()} to {fundamentals['period_end_date'].max()}")
    
    # Standardize columns
    print("\n[3/5] Standardizing columns...")
    fundamentals = standardize_columns(fundamentals)
    
    # Compute derived metrics
    print("\n[4/5] Computing derived metrics...")
    fundamentals = compute_derived_metrics(fundamentals)
    
    # Get list of derived metric columns
    derived_cols = [c for c in DERIVED_METRICS if c in fundamentals.columns]
    base_cols = [c for c in COLUMN_MAP.values() if c in fundamentals.columns and c not in derived_cols]
    all_feature_cols = base_cols + derived_cols
    print(f"  Base metrics: {len(base_cols)}")
    print(f"  Derived metrics: {len(derived_cols)}")
    print(f"  Total feature columns: {len(all_feature_cols)}")
    
    # Group fundamentals by symbol for fast lookup
    print("\n[5/5] Building point-in-time features for each event...")
    fundamentals_by_symbol = {
        sym: grp.sort_values("period_end_date").reset_index(drop=True)
        for sym, grp in fundamentals.groupby("symbol")
    }
    
    rows = []
    stats = {
        "matched": 0,
        "no_fundamentals": 0,
        "no_available_before_cutoff": 0,
    }
    
    for _, event in events.iterrows():
        symbol = event["symbol"]
        cutoff = event["feature_cutoff_date"]
        event_key = event["event_key"]
        
        if symbol not in fundamentals_by_symbol:
            stats["no_fundamentals"] += 1
            rows.append({
                "event_key": event_key,
                "fundamental_period_end_date": pd.NaT,
                "fundamental_estimated_available_date": pd.NaT,
                "fundamental_pit_status": "unavailable",
                "fundamental_lag_days": np.nan,
            })
            continue
        
        sym_fund = fundamentals_by_symbol[symbol]
        latest = find_latest_fundamental_before_cutoff(sym_fund, cutoff, event["period_ended"])
        
        if latest is None:
            stats["no_available_before_cutoff"] += 1
            rows.append({
                "event_key": event_key,
                "fundamental_period_end_date": pd.NaT,
                "fundamental_estimated_available_date": pd.NaT,
                "fundamental_pit_status": "unavailable",
                "fundamental_lag_days": np.nan,
            })
            continue
        
        stats["matched"] += 1
        
        # Calculate lag from estimated available date to feature cutoff
        estimated_avail = latest["estimated_available_date"]
        lag_days = (cutoff - estimated_avail).days
        
        # Build row with all available features
        row = {
            "event_key": event_key,
            "fundamental_period_end_date": latest["period_end_date"],
            "fundamental_estimated_available_date": estimated_avail,
            "fundamental_pit_status": "estimated_conservative",
            "fundamental_lag_days": lag_days,
        }
        
        # Add all available feature columns
        for col in all_feature_cols:
            if col in latest.index:
                row[f"fund_{col}"] = latest[col]
            else:
                row[f"fund_{col}"] = np.nan
        
        rows.append(row)
    
    print(f"  Matched: {stats['matched']}")
    print(f"  No fundamentals for symbol: {stats['no_fundamentals']}")
    print(f"  No data before cutoff: {stats['no_available_before_cutoff']}")
    
    # Build output
    fund_features = pd.DataFrame(rows)
    
    # Merge with events to get full dataset
    output = events.merge(fund_features, on="event_key", how="left")
    
    # Reorder columns: metadata + market features + fundamental features + labels
    meta_cols = [
        "event_key", "symbol", "company_name", "period_ended", "fiscal_quarter",
        "result_announcement_datetime", "announcement_session_type",
        "feature_cutoff_date", "reaction_start_date",
        "reaction_class", "quality_flags",
        "price_history_days", "benchmark_history_days",
    ]
    fund_cols = [c for c in output.columns if c.startswith("fund_")]
    market_feature_cols = [c for c in output.columns 
                           if c not in meta_cols and not c.startswith("fund_") and not c.endswith("_after")]
    label_cols = [c for c in output.columns if c.endswith("_after")]
    
    ordered = meta_cols + sorted(market_feature_cols) + sorted(fund_cols) + sorted(label_cols)
    output = output[ordered]
    
    # Save
    output.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE} ({len(output)} rows, {len(output.columns)} columns)")
    
    # Quality report
    quality_df = output[
        meta_cols + [
            "fundamental_period_end_date",
            "fundamental_estimated_available_date",
            "fundamental_pit_status",
            "fundamental_lag_days"
        ]
    ].copy()
    quality_df.to_csv(QUALITY_REPORT, index=False)
    print(f"Quality report: {QUALITY_REPORT}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FUNDAMENTAL FEATURE SUMMARY")
    print("=" * 70)
    print(f"Events with fundamental data: {stats['matched']}/{len(events)} ({stats['matched']/len(events)*100:.1f}%)")
    print(f"Events without (unavailable): {stats['no_fundamentals'] + stats['no_available_before_cutoff']}")
    print(f"Fundamental columns added: {len(fund_cols)}")
    
    if stats['matched'] > 0:
        with_data = output[output["fundamental_pit_status"] == "estimated_conservative"]
        print(f"\nFundamental lag stats (days from estimated_available_date to feature_cutoff):")
        print(f"  Mean: {with_data['fundamental_lag_days'].mean():.1f}")
        print(f"  Median: {with_data['fundamental_lag_days'].median():.1f}")
        print(f"  Max: {with_data['fundamental_lag_days'].max()}")
        print(f"  Min: {with_data['fundamental_lag_days'].min()}")
        
        print(f"\nPIT Status distribution:")
        print(output["fundamental_pit_status"].value_counts().to_string())
    
    print("\nFeature availability (non-null % for events with fundamentals):")
    for col in sorted(fund_cols):
        if stats['matched'] > 0:
            with_data = output[output["fundamental_pit_status"] == "estimated_conservative"]
            pct = with_data[col].notna().mean() * 100
            print(f"  {col}: {pct:.1f}%")
    
    print("\nDone!")
    return output


if __name__ == "__main__":
    build_fundamental_features()