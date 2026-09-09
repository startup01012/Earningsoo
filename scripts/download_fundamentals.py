"""
Download Fundamental Data from Yahoo Finance (yfinance)

Downloads quarterly financial statements (income statement, balance sheet, cash flow)
for NIFTY 50 symbols and saves as normalized parquet files.

Output: data/raw/fundamentals/{symbol}.parquet
"""

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# ============================================================
# CONFIG
# ============================================================

REFERENCE_FILE = Path("data/reference/nifty50_clean.csv")
OUTPUT_DIR = Path("data/raw/fundamentals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY = 1.0  # seconds between requests
MAX_RETRIES = 3

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_reference() -> pd.DataFrame:
    """Load NIFTY 50 reference data."""
    ref = pd.read_csv(REFERENCE_FILE)
    ref["symbol"] = ref["symbol"].astype(str).str.strip()
    return ref


def download_fundamentals_yfinance(symbol: str, max_retries: int = MAX_RETRIES) -> Optional[dict]:
    """Download quarterly financial statements from yfinance."""
    yf_symbol = f"{symbol}.NS"
    
    for attempt in range(1, max_retries + 1):
        try:
            ticker = yf.Ticker(yf_symbol)
            
            # Get quarterly financial statements
            # yfinance returns DataFrames with dates as columns (quarterly periods)
            income_stmt = ticker.quarterly_income_stmt
            balance_sheet = ticker.quarterly_balance_sheet
            cash_flow = ticker.quarterly_cashflow
            
            # Also get trailing annual data for context
            annual_income = ticker.income_stmt
            annual_balance = ticker.balance_sheet
            annual_cashflow = ticker.cashflow
            
            return {
                "quarterly_income_stmt": income_stmt,
                "quarterly_balance_sheet": balance_sheet,
                "quarterly_cashflow": cash_flow,
                "annual_income_stmt": annual_income,
                "annual_balance_sheet": annual_balance,
                "annual_cashflow": annual_cashflow,
            }
            
        except Exception as e:
            print(f"  [ATTEMPT {attempt}/{max_retries}] {symbol}: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
    
    return None


def normalize_financial_statement(df: pd.DataFrame, statement_type: str) -> pd.DataFrame:
    """Normalize a financial statement DataFrame from yfinance.
    
    yfinance returns DataFrames with:
    - Index: line items (e.g., 'Total Revenue', 'Net Income')
    - Columns: period end dates (quarterly or annual)
    
    We transpose to have dates as rows and line items as columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Transpose so dates are rows
    df = df.T
    
    # Reset index to get date column
    df = df.reset_index()
    df = df.rename(columns={"index": "period_end_date"})
    
    # Convert period_end_date to datetime
    df["period_end_date"] = pd.to_datetime(df["period_end_date"], errors="coerce")
    
    # Add statement type prefix to columns (except date)
    cols_to_rename = {c: f"{statement_type}_{c}" for c in df.columns if c != "period_end_date"}
    df = df.rename(columns=cols_to_rename)
    
    return df


def merge_fundamental_data(data: dict) -> pd.DataFrame:
    """Merge all financial statements into a single DataFrame."""
    statements = []
    
    # Quarterly statements (higher frequency, more detailed)
    for stmt_name, prefix in [
        ("quarterly_income_stmt", "q_is"),
        ("quarterly_balance_sheet", "q_bs"),
        ("quarterly_cashflow", "q_cf"),
    ]:
        if stmt_name in data and data[stmt_name] is not None and not data[stmt_name].empty:
            norm = normalize_financial_statement(data[stmt_name], prefix)
            if not norm.empty:
                statements.append(norm)
    
    # Annual statements (for filling gaps)
    for stmt_name, prefix in [
        ("annual_income_stmt", "a_is"),
        ("annual_balance_sheet", "a_bs"),
        ("annual_cashflow", "a_cf"),
    ]:
        if stmt_name in data and data[stmt_name] is not None and not data[stmt_name].empty:
            norm = normalize_financial_statement(data[stmt_name], prefix)
            if not norm.empty:
                statements.append(norm)
    
    if not statements:
        return pd.DataFrame()
    
    # Merge all statements on period_end_date
    merged = statements[0]
    for stmt in statements[1:]:
        merged = pd.merge(merged, stmt, on="period_end_date", how="outer")
    
    # Sort by date
    merged = merged.sort_values("period_end_date").reset_index(drop=True)
    
    return merged


def save_fundamentals(df: pd.DataFrame, symbol: str):
    """Save fundamentals DataFrame to parquet."""
    output_file = OUTPUT_DIR / f"{symbol}.parquet"
    df.to_parquet(output_file, index=False, compression="snappy")


def download_all_fundamentals():
    """Main function to download fundamentals for all NIFTY 50 symbols."""
    print("=" * 70)
    print("DOWNLOAD FUNDAMENTAL DATA FROM YAHOO FINANCE")
    print("=" * 70)
    print(f"Output dir: {OUTPUT_DIR}")
    print()
    
    ref = load_reference()
    symbols = ref["symbol"].tolist()
    print(f"Total symbols: {len(symbols)}")
    
    success_count = 0
    failed = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Downloading {symbol}...")
        
        data = download_fundamentals_yfinance(symbol)
        
        if data is None:
            print(f"  [FAILED] No data returned")
            failed.append(symbol)
            continue
        
        merged = merge_fundamental_data(data)
        
        if merged.empty:
            print(f"  [WARNING] No fundamental data after normalization")
            failed.append(symbol)
            continue
        
        print(f"  [OK] {len(merged)} periods | {merged['period_end_date'].min().date()} to {merged['period_end_date'].max().date()}")
        print(f"       Columns: {len(merged.columns)}")
        
        save_fundamentals(merged, symbol)
        success_count += 1
        
        # Rate limiting
        time.sleep(REQUEST_DELAY)
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed symbols: {failed}")


def verify_fundamentals():
    """Verify downloaded fundamental data."""
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    files = sorted(OUTPUT_DIR.glob("*.parquet"))
    print(f"Total parquet files: {len(files)}")
    
    for f in files[:5]:
        df = pd.read_parquet(f)
        print(f"\n{f.stem}: {len(df)} rows, {len(df.columns)} cols")
        print(f"  Date range: {df['period_end_date'].min()} to {df['period_end_date'].max()}")
        # Show some key columns
        key_cols = [c for c in df.columns if any(k in c.lower() for k in ['revenue', 'income', 'profit', 'eps', 'asset', 'equity', 'debt', 'cash'])]
        print(f"  Key columns: {key_cols[:10]}")


if __name__ == "__main__":
    download_all_fundamentals()
    verify_fundamentals()