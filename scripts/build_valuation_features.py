"""
Build Point-in-Time Valuation Features

For each earnings event, computes valuation metrics using:
- Market price at/before feature_cutoff_date
- Latest fundamental data available at/before feature_cutoff_date

Output: data/processed/earnings_event_valuation_features.parquet
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

# Input: ML dataset with fundamental features
ML_INPUT = Path("data/processed/earnings_event_fundamental_features.parquet")

# Price data for market cap / price at cutoff
PRICES_DIR = Path("data/raw/prices")

OUTPUT_FILE = Path("data/processed/earnings_event_valuation_features.parquet")
QUALITY_REPORT = Path("data/processed/earnings_event_valuation_features_quality_report.csv")

for p in [OUTPUT_FILE, QUALITY_REPORT]:
    p.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================

def load_prices_for_valuation() -> pd.DataFrame:
    """Load price data and prepare for valuation lookups."""
    files = sorted(PRICES_DIR.glob("*.parquet"))
    all_dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_convert(None)
        all_dfs.append(df)
    prices = pd.concat(all_dfs, ignore_index=True)
    prices = prices.dropna(subset=["date", "symbol", "close"])
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    return prices


def get_price_at_or_before(prices: pd.DataFrame, symbol: str, cutoff_date: pd.Timestamp) -> Optional[float]:
    """Get the latest close price for symbol on or before cutoff_date."""
    sym_prices = prices[(prices["symbol"] == symbol) & (prices["date"] <= cutoff_date)]
    if sym_prices.empty:
        return None
    return sym_prices.iloc[-1]["close"]


def get_shares_outstanding(fund_row: pd.Series) -> Optional[float]:
    """Get shares outstanding from fundamental data (quarterly preferred, then annual)."""
    # Try quarterly first
    for col in ["fund_shares_outstanding", "fund_shares_diluted", "fund_shares_basic"]:
        if col in fund_row and pd.notna(fund_row[col]):
            return fund_row[col]
    # Try annual
    for col in ["fund_a_shares_outstanding"]:
        if col in fund_row and pd.notna(fund_row[col]):
            return fund_row[col]
    return None


def get_book_value_per_share(fund_row: pd.Series) -> Optional[float]:
    """Calculate book value per share = equity / shares_outstanding."""
    equity = None
    for col in ["fund_equity", "fund_total_equity", "fund_a_equity"]:
        if col in fund_row and pd.notna(fund_row[col]):
            equity = fund_row[col]
            break
    
    shares = get_shares_outstanding(fund_row)
    
    if equity is not None and shares is not None and shares > 0:
        return equity / shares
    return None


def get_eps(fund_row: pd.Series) -> Optional[float]:
    """Get EPS (diluted preferred)."""
    for col in ["fund_eps_diluted", "fund_eps_basic", "fund_a_eps_diluted"]:
        if col in fund_row and pd.notna(fund_row[col]):
            return fund_row[col]
    return None


def get_ebitda(fund_row: pd.Series) -> Optional[float]:
    """Get EBITDA."""
    for col in ["fund_ebitda", "fund_normalized_ebitda"]:
        if col in fund_row and pd.notna(fund_row[col]):
            return fund_row[col]
    return None


def get_total_debt(fund_row: pd.Series) -> Optional[float]:
    """Get total debt."""
    for col in ["fund_total_debt", "fund_a_total_debt"]:
        if col in fund_row and pd.notna(fund_row[col]):
            return fund_row[col]
    return None


def get_cash(fund_row: pd.Series) -> Optional[float]:
    """Get cash and equivalents."""
    for col in ["fund_cash_and_sti", "fund_cash", "fund_a_cash"]:
        if col in fund_row and pd.notna(fund_row[col]):
            return fund_row[col]
    return None


def safe_divide(num: float, den: float) -> float:
    """Safe division."""
    if den is None or den == 0 or np.isnan(den) or np.isinf(den):
        return np.nan
    if num is None or np.isnan(num) or np.isinf(num):
        return np.nan
    return num / den


def compute_valuation_features(row: pd.Series, prices: pd.DataFrame) -> dict:
    """Compute all valuation features for a single event."""
    symbol = row["symbol"]
    cutoff = row["feature_cutoff_date"]
    
    # Get price at cutoff
    price = get_price_at_or_before(prices, symbol, cutoff)
    if price is None:
        return {f"val_{k}": np.nan for k in [
            "pe", "pb", "earnings_yield", "market_cap", "ev", "ev_ebitda",
            "ps", "pcf"
        ]}
    
    # Get fundamental data
    shares = get_shares_outstanding(row)
    bvps = get_book_value_per_share(row)
    eps = get_eps(row)
    ebitda = get_ebitda(row)
    total_debt = get_total_debt(row)
    cash = get_cash(row)
    
    features = {}
    
    # Market Cap
    if shares is not None and shares > 0:
        features["val_market_cap"] = price * shares
    else:
        features["val_market_cap"] = np.nan
    
    # P/E
    features["val_pe"] = safe_divide(price, eps)
    
    # Earnings Yield
    pe = features["val_pe"]
    features["val_earnings_yield"] = safe_divide(1, pe) if pd.notna(pe) else np.nan
    
    # P/B
    features["val_pb"] = safe_divide(price, bvps)
    
    # P/S (Price/Sales) - need revenue
    revenue = row.get("fund_revenue", np.nan)
    if pd.notna(revenue) and shares is not None and shares > 0 and revenue > 0:
        ps = (price * shares) / revenue
        features["val_ps"] = ps
    else:
        features["val_ps"] = np.nan
    
    # EV = Market Cap + Total Debt - Cash
    mcap = features["val_market_cap"]
    if pd.notna(mcap) and total_debt is not None and cash is not None:
        features["val_ev"] = mcap + total_debt - cash
    else:
        features["val_ev"] = np.nan
    
    # EV/EBITDA
    ev = features["val_ev"]
    features["val_ev_ebitda"] = safe_divide(ev, ebitda)
    
    # P/CF (Price/Operating Cash Flow)
    ocf = row.get("fund_operating_cash_flow", np.nan)
    if pd.notna(ocf) and shares is not None and shares > 0 and ocf > 0:
        features["val_pcf"] = (price * shares) / ocf
    else:
        features["val_pcf"] = np.nan
    
    # Dividend yield - not available from our data, DO NOT CREATE THIS FEATURE
    # features["val_dividend_yield"] = np.nan  # REMOVED: 100% NaN, no valid historical source
    
    return features


def build_valuation_features():
    """Main pipeline."""
    print("=" * 70)
    print("BUILD POINT-IN-TIME VALUATION FEATURES")
    print("=" * 70)
    
    # Load ML dataset with fundamentals
    print("\n[1/4] Loading ML dataset with fundamentals...")
    ml_data = pd.read_parquet(ML_INPUT)
    print(f"  Rows: {len(ml_data)}")
    print(f"  Columns: {len(ml_data.columns)}")
    print(f"  Events with fundamentals: {ml_data['fundamental_pit_status'].eq('estimated_conservative').sum()}")
    
    # Load prices
    print("\n[2/4] Loading price data...")
    prices = load_prices_for_valuation()
    print(f"  Price rows: {len(prices):,}")
    print(f"  Symbols: {prices['symbol'].nunique()}")
    print(f"  Date range: {prices['date'].min()} to {prices['date'].max()}")
    
    # Compute valuation features
    print("\n[3/4] Computing valuation features...")
    val_rows = []
    for _, row in ml_data.iterrows():
        val_feats = compute_valuation_features(row, prices)
        val_feats["event_key"] = row["event_key"]
        val_rows.append(val_feats)
    
    val_df = pd.DataFrame(val_rows)
    
    # Merge back
    print("\n[4/4] Merging and saving...")
    output = ml_data.merge(val_df, on="event_key", how="left")
    
    # Reorder columns
    val_cols = [c for c in output.columns if c.startswith("val_")]
    print(f"  Valuation columns added: {len(val_cols)}")
    
    # Save
    output.to_parquet(OUTPUT_FILE, index=False)
    print(f"  Saved: {OUTPUT_FILE} ({len(output)} rows, {len(output.columns)} columns)")
    
    # Quality report
    quality_cols = ["event_key", "symbol", "feature_cutoff_date"] + val_cols
    output[quality_cols].to_csv(QUALITY_REPORT, index=False)
    print(f"  Quality report: {QUALITY_REPORT}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALUATION FEATURE SUMMARY")
    print("=" * 70)
    
    for col in sorted(val_cols):
        pct = output[col].notna().mean() * 100
        print(f"  {col}: {pct:.1f}% non-null")
    
    print("\nDone!")
    return output


if __name__ == "__main__":
    build_valuation_features()