"""
Download Historical Price Data from Yahoo Finance (yfinance)

This script downloads daily OHLCV data for NIFTY 50 symbols from 2016-01-01
to fill the gap before NSE Bhavcopy data (which starts ~2024).

Output format matches the NSE Bhavcopy parquet files for seamless integration.
"""

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from download_historical_prices import HEADERS, OUTPUT_DIR

# ============================================================
# CONFIG
# ============================================================

# Start from 2016-01-01 (before NSE Bhavcopy availability)
YF_START_DATE = "2016-01-01"

# End date: day before NSE Bhavcopy data starts (2024-07-08)
# We'll download up to 2024-07-07 to avoid overlap
YF_END_DATE = "2024-07-07"

# Reference file for symbol mapping and ISIN
REFERENCE_FILE = Path("data/reference/nifty50_clean.csv")

# Additional symbols to download (benchmarks)
BENCHMARK_SYMBOLS = {
    "NIFTYBEES": "NIFTYBEES.NS",  # NIFTY 50 ETF
    "NIFTY50": "^NSEI",           # NIFTY 50 Index
}

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3

# Column mapping from yfinance to our schema
YF_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}

# Required output columns (matching NSE Bhavcopy schema)
REQUIRED_OUTPUT_COLUMNS = [
    "date",
    "symbol",
    "isin",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "traded_value",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_reference() -> pd.DataFrame:
    """Load NIFTY 50 reference data with ISIN mapping."""
    ref = pd.read_csv(REFERENCE_FILE)
    ref["symbol"] = ref["symbol"].astype(str).str.strip()
    ref["isin"] = ref["isin"].astype(str).str.strip()
    # Create symbol -> isin mapping
    symbol_to_isin = dict(zip(ref["symbol"], ref["isin"]))
    return ref, symbol_to_isin


def download_symbol_yfinance(symbol: str, start: str, end: str, max_retries: int = MAX_RETRIES) -> Optional[pd.DataFrame]:
    """Download historical data for one symbol from yfinance."""
    yf_symbol = f"{symbol}.NS"
    
    for attempt in range(1, max_retries + 1):
        try:
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(start=start, end=end, auto_adjust=False)
            
            if hist.empty:
                return None
            
            # Reset index to get date as column
            hist = hist.reset_index()
            
            # Rename columns
            hist = hist.rename(columns={
                "Date": "date",
                **YF_COLUMN_MAP
            })
            
            # Keep only needed columns from yfinance
            available_cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in hist.columns]
            hist = hist[available_cols].copy()
            
            return hist
            
        except Exception as e:
            print(f"  [ATTEMPT {attempt}/{max_retries}] {symbol}: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
    
    return None


def compute_prev_close_and_traded_value(df: pd.DataFrame) -> pd.DataFrame:
    """Compute prev_close and traded_value columns."""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # prev_close = previous day's close
    df["prev_close"] = df["close"].shift(1)
    
    # traded_value = close * volume (approximation)
    df["traded_value"] = df["close"] * df["volume"]
    
    return df


def add_symbol_and_isin(df: pd.DataFrame, symbol: str, isin: str) -> pd.DataFrame:
    """Add symbol and ISIN columns."""
    df = df.copy()
    df["symbol"] = symbol
    df["isin"] = isin
    return df


def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct data types matching NSE Bhavcopy schema."""
    df = df.copy()
    
    # Date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Numeric columns
    numeric_cols = ["open", "high", "low", "close", "prev_close", "volume", "traded_value"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # String columns
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["isin"] = df["isin"].astype(str).str.strip()
    
    # Remove rows with missing critical data
    df = df.dropna(subset=["date", "symbol", "close"])
    
    return df


def save_daily_files(df: pd.DataFrame) -> dict:
    """Save DataFrame as daily parquet files (one per trading day)."""
    stats = {"saved": 0, "skipped": 0, "failed": 0}
    
    # Group by date
    for date_val, group in df.groupby("date"):
        date_str = date_val.strftime("%Y%m%d")
        output_file = OUTPUT_DIR / f"{date_str}.parquet"
        
        # Skip if already exists (resume support)
        if output_file.exists():
            stats["skipped"] += 1
            continue
        
        try:
            # Select and order columns
            group = group[REQUIRED_OUTPUT_COLUMNS].copy()
            
            group.to_parquet(output_file, index=False, compression="snappy")
            stats["saved"] += 1
        except Exception as e:
            print(f"  [ERROR] Saving {date_str}: {e}")
            stats["failed"] += 1
    
    return stats


def download_all_symbols():
    """Main function to download all NIFTY 50 symbols + benchmarks from yfinance."""
    print("=" * 70)
    print("DOWNLOAD HISTORICAL PRICES FROM YAHOO FINANCE")
    print("=" * 70)
    print(f"Date range: {YF_START_DATE} to {YF_END_DATE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()
    
    # Load reference data
    ref, symbol_to_isin = load_reference()
    symbols = ref["symbol"].tolist()
    
    # Add benchmark symbols
    benchmark_list = list(BENCHMARK_SYMBOLS.items())
    all_symbols = symbols + [s[0] for s in benchmark_list]
    symbol_to_yf = {s: f"{s}.NS" for s in symbols}
    symbol_to_yf.update({k: v for k, v in benchmark_list})
    symbol_to_isin.update({k: "" for k, _ in benchmark_list})  # No ISIN for benchmarks
    
    print(f"Total symbols (incl. benchmarks): {len(all_symbols)}")
    
    all_data = []
    
    for i, symbol in enumerate(all_symbols, 1):
        isin = symbol_to_isin.get(symbol, "")
        yf_symbol = symbol_to_yf.get(symbol, f"{symbol}.NS")
        print(f"[{i}/{len(all_symbols)}] Downloading {symbol} (YF: {yf_symbol}, ISIN: {isin})...")
        
        hist = download_symbol_yfinance(yf_symbol, YF_START_DATE, YF_END_DATE)
        
        if hist is None or hist.empty:
            print(f"  [WARNING] No data for {symbol}")
            continue
        
        # Process
        hist = compute_prev_close_and_traded_value(hist)
        hist = add_symbol_and_isin(hist, symbol, isin)
        hist = normalize_dtypes(hist)
        
        print(f"  [OK] {len(hist)} rows | {hist['date'].min().date()} to {hist['date'].max().date()}")
        all_data.append(hist)
        
        # Rate limiting
        time.sleep(REQUEST_DELAY)
    
    if not all_data:
        print("\n[ERROR] No data downloaded!")
        return
    
    # Combine all symbols
    print("\nCombining all symbols...")
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total rows: {len(combined):,}")
    print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    
    # Save as daily files
    print("\nSaving daily parquet files...")
    stats = save_daily_files(combined)
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    for key, value in stats.items():
        print(f"  {key:10}: {value:,}")


def verify_downloaded_data():
    """Verify the downloaded data covers expected range."""
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    files = sorted(OUTPUT_DIR.glob("*.parquet"))
    print(f"Total parquet files: {len(files)}")
    
    if not files:
        print("No files found!")
        return
    
    # Check date range
    dates = [f.stem for f in files]
    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Load a sample to verify schema
    sample = pd.read_parquet(files[0])
    print(f"\nSample file ({files[0].name}):")
    print(f"  Rows: {len(sample)}")
    print(f"  Columns: {sample.columns.tolist()}")
    print(f"  Dtypes:\n{sample.dtypes}")
    print(f"\n  Head:\n{sample.head()}")


if __name__ == "__main__":
    download_all_symbols()
    verify_downloaded_data()