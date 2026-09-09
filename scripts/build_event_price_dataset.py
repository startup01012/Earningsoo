"""
Build Event → Price ML Dataset

Joins canonical earnings events with historical market prices to create
leakage-safe pre-event features and post-event reaction labels.

One row per canonical earnings event (symbol + period_ended).
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

EVENTS_INPUT = Path("data/processed/earnings_events.parquet")
PRICES_DIR = Path("data/raw/prices")
REFERENCE_FILE = Path("data/reference/nifty50_clean.csv")

EVENTS_OUTPUT = Path("data/processed/earnings_event_market_features.parquet")
QUALITY_REPORT_OUTPUT = Path("data/processed/earnings_event_market_features_quality_report.csv")

for p in [EVENTS_OUTPUT, QUALITY_REPORT_OUTPUT]:
    p.parent.mkdir(parents=True, exist_ok=True)

# Benchmark: use NIFTYBEES (most liquid NIFTY 50 ETF)
BENCHMARK_SYMBOL = "NIFTYBEES"

# Market hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Feature windows (trading days)
RETURN_WINDOWS = [1, 3, 5, 10, 20, 60]
VOLATILITY_WINDOWS = [5, 20, 60]
VOLUME_WINDOWS = [5, 20]
MA_WINDOWS = [20, 50, 200]
DRAWDOWN_WINDOWS = [20, 60, 252]

# Label windows (trading days after reaction start)
LABEL_WINDOWS = [1, 3, 5]

# Minimum history requirements
MIN_HISTORY_FOR_FEATURES = {
    "return_1d": 2,
    "return_3d": 4,
    "return_5d": 6,
    "return_10d": 11,
    "return_20d": 21,
    "return_60d": 61,
    "volatility_5d": 6,
    "volatility_20d": 21,
    "volatility_60d": 61,
    "volume_ratio_5d": 6,
    "volume_ratio_20d": 21,
    "distance_from_20dma": 21,
    "distance_from_50dma": 51,
    "distance_from_200dma": 201,
    "drawdown_from_20d_high": 21,
    "drawdown_from_60d_high": 61,
    "drawdown_from_252d_high": 253,
}

# ============================================================
# UTILITIES
# ============================================================

def load_all_prices() -> pd.DataFrame:
    """Load and concatenate all daily price files."""
    files = sorted(PRICES_DIR.glob("*.parquet"))
    all_dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Normalize timezone: convert tz-aware to tz-naive (UTC -> local -> naive)
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_convert(None)
        all_dfs.append(df)
    prices = pd.concat(all_dfs, ignore_index=True)
    prices = prices.dropna(subset=["date", "symbol", "close"])
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    return prices


def get_trading_days(prices: pd.DataFrame) -> np.ndarray:
    """Extract sorted unique trading days from price data."""
    return np.sort(prices["date"].dt.date.unique())


def build_trading_day_lookup(trading_days: np.ndarray) -> dict:
    """Build lookup maps for trading day navigation."""
    # Map date -> index in trading_days array
    date_to_idx = {d: i for i, d in enumerate(trading_days)}
    # Map index -> date
    idx_to_date = {i: d for i, d in enumerate(trading_days)}
    return date_to_idx, idx_to_date


def classify_announcement_session(ann_dt: pd.Timestamp, trading_days: np.ndarray = None) -> str:
    """
    Classify announcement timing relative to market hours.

    Returns:
        'pre_market'    - before 9:15 AM on a trading day
        'during_market' - 9:15 AM to 3:30 PM on a trading day
        'post_market'   - after 3:30 PM on a trading day
        'non_trading_day' - announcement on weekend/holiday
        'unknown'       - time is 00:00:00 or NaT (treat conservatively)
    """
    if pd.isna(ann_dt):
        return "unknown"

    # Check if time is midnight (likely missing time data)
    if ann_dt.hour == 0 and ann_dt.minute == 0 and ann_dt.second == 0:
        return "unknown"

    ann_date = ann_dt.date()

    # Check if announcement date is a trading day
    if trading_days is not None and ann_date not in trading_days:
        return "non_trading_day"

    ann_time_minutes = ann_dt.hour * 60 + ann_dt.minute
    market_open_minutes = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE
    market_close_minutes = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE

    if ann_time_minutes < market_open_minutes:
        return "pre_market"
    elif ann_time_minutes <= market_close_minutes:
        return "during_market"
    else:
        return "post_market"


def get_feature_cutoff_date(
    ann_dt: pd.Timestamp,
    trading_days: np.ndarray,
    date_to_idx: dict,
) -> Optional[pd.Timestamp]:
    """
    Determine the latest eligible trading date for pre-event features.

    Rules:
    - pre_market:         use previous trading day
    - during_market:      use previous trading day (conservative, no intraday precision)
    - post_market:        use announcement date (if trading day) else previous trading day
    - non_trading_day:    use last trading day before announcement
    - unknown:            use previous trading day (conservative)
    """
    if pd.isna(ann_dt):
        return None

    ann_date = ann_dt.date()
    session = classify_announcement_session(ann_dt, trading_days)

    # Find the trading day index for announcement date (or prior)
    if ann_date in date_to_idx:
        ann_idx = date_to_idx[ann_date]
        is_trading_day = True
    else:
        # Weekend/holiday: find last trading day before
        prior_idx = None
        for i, td in enumerate(trading_days):
            if td <= ann_date:
                prior_idx = i
            else:
                break
        if prior_idx is None:
            return None
        ann_idx = prior_idx
        is_trading_day = False

    if session == "post_market" and is_trading_day:
        # After close on trading day: use this day's close
        cutoff_idx = ann_idx
    else:
        # Pre-market, during market, non_trading_day, unknown: use previous trading day
        cutoff_idx = ann_idx - 1

    if cutoff_idx < 0:
        return None

    return pd.Timestamp(trading_days[cutoff_idx])


def get_reaction_start_date(
    ann_dt: pd.Timestamp,
    trading_days: np.ndarray,
    date_to_idx: dict,
) -> Optional[pd.Timestamp]:
    """
    Determine the first trading day for post-event reaction measurement.

    Rules:
    - pre_market:         reaction starts on announcement day (if trading day) else next trading day
    - during_market:      reaction starts on next trading day (conservative)
    - post_market:        reaction starts on next trading day
    - non_trading_day:    reaction starts on next trading day after announcement
    - unknown:            reaction starts on next trading day (conservative)
    """
    if pd.isna(ann_dt):
        return None

    ann_date = ann_dt.date()
    session = classify_announcement_session(ann_dt, trading_days)

    # Find position in trading days
    if ann_date in date_to_idx:
        ann_idx = date_to_idx[ann_date]
        is_trading_day = True
    else:
        # Weekend/holiday: find last trading day before
        prior_idx = None
        for i, td in enumerate(trading_days):
            if td <= ann_date:
                prior_idx = i
            else:
                break
        if prior_idx is None:
            return None
        ann_idx = prior_idx
        is_trading_day = False

    if session == "pre_market" and is_trading_day:
        # Before open on trading day: reaction starts same day
        reaction_idx = ann_idx
    else:
        # During market, post-market, non_trading_day, unknown: next trading day
        reaction_idx = ann_idx + 1

    if reaction_idx >= len(trading_days):
        return None

    return pd.Timestamp(trading_days[reaction_idx])


def calculate_returns(prices: pd.Series, windows: list) -> dict:
    """Calculate returns for multiple windows. prices indexed by date (ascending)."""
    result = {}
    for w in windows:
        if len(prices) >= w + 1:
            # Return over w days: (price_t / price_{t-w}) - 1
            result[f"return_{w}d"] = prices.iloc[-1] / prices.iloc[-w-1] - 1
        else:
            result[f"return_{w}d"] = np.nan
    return result


def calculate_volatility(returns: pd.Series, windows: list) -> dict:
    """Calculate rolling volatility (std of returns) for multiple windows."""
    result = {}
    for w in windows:
        if len(returns) >= w:
            result[f"volatility_{w}d"] = returns.iloc[-w:].std() * np.sqrt(252)  # Annualized
        else:
            result[f"volatility_{w}d"] = np.nan
    return result


def calculate_volume_features(volumes: pd.Series, windows: list) -> dict:
    """Calculate volume ratio features."""
    result = {}
    if len(volumes) == 0:
        for w in windows:
            result[f"volume_ratio_{w}d"] = np.nan
            result[f"volume_zscore_{w}d"] = np.nan
        return result

    current_vol = volumes.iloc[-1]
    for w in windows:
        if len(volumes) >= w:
            hist_vol = volumes.iloc[-w-1:-1]  # Exclude current day
            if len(hist_vol) > 0 and hist_vol.mean() > 0:
                result[f"volume_ratio_{w}d"] = current_vol / hist_vol.mean()
            else:
                result[f"volume_ratio_{w}d"] = np.nan

            if len(hist_vol) > 1 and hist_vol.std() > 0:
                result[f"volume_zscore_{w}d"] = (current_vol - hist_vol.mean()) / hist_vol.std()
            else:
                result[f"volume_zscore_{w}d"] = np.nan
        else:
            result[f"volume_ratio_{w}d"] = np.nan
            result[f"volume_zscore_{w}d"] = np.nan
    return result


def calculate_ma_distance(prices: pd.Series, windows: list) -> dict:
    """Calculate distance from moving averages."""
    result = {}
    current_price = prices.iloc[-1]
    for w in windows:
        if len(prices) >= w:
            ma = prices.iloc[-w:].mean()
            if ma > 0:
                result[f"distance_from_{w}dma"] = (current_price - ma) / ma
            else:
                result[f"distance_from_{w}dma"] = np.nan
        else:
            result[f"distance_from_{w}dma"] = np.nan
    return result


def calculate_drawdown(prices: pd.Series, windows: list) -> dict:
    """Calculate drawdown from recent highs."""
    result = {}
    current_price = prices.iloc[-1]
    for w in windows:
        if len(prices) >= w:
            high = prices.iloc[-w:].max()
            if high > 0:
                result[f"drawdown_from_{w}d_high"] = (current_price - high) / high
            else:
                result[f"drawdown_from_{w}d_high"] = np.nan
        else:
            result[f"drawdown_from_{w}d_high"] = np.nan
    return result


def calculate_relative_returns(
    stock_returns: dict,
    bench_returns: dict,
    windows: list
) -> dict:
    """Calculate benchmark-relative returns."""
    result = {}
    for w in windows:
        stock_key = f"return_{w}d"
        bench_key = f"return_{w}d"
        if stock_key in stock_returns and bench_key in bench_returns:
            sr = stock_returns[stock_key]
            br = bench_returns[bench_key]
            if not (np.isnan(sr) or np.isnan(br)):
                result[f"relative_return_{w}d"] = sr - br
            else:
                result[f"relative_return_{w}d"] = np.nan
        else:
            result[f"relative_return_{w}d"] = np.nan
    return result


def calculate_abnormal_returns(
    stock_prices: pd.Series,
    bench_prices: pd.Series,
    label_windows: list,
    reaction_start_idx: int,
    trading_days: np.ndarray,
) -> dict:
    """
    Calculate abnormal returns for label windows.
    Uses forward returns from reaction_start_date.
    """
    result = {}
    for w in label_windows:
        end_idx = reaction_start_idx + w
        if end_idx < len(stock_prices) and end_idx < len(bench_prices):
            stock_ret = stock_prices.iloc[end_idx] / stock_prices.iloc[reaction_start_idx] - 1
            bench_ret = bench_prices.iloc[end_idx] / bench_prices.iloc[reaction_start_idx] - 1
            result[f"abnormal_return_{w}d"] = stock_ret - bench_ret
            result[f"return_{w}d_after"] = stock_ret
            result[f"benchmark_return_{w}d_after"] = bench_ret
        else:
            result[f"abnormal_return_{w}d"] = np.nan
            result[f"return_{w}d_after"] = np.nan
            result[f"benchmark_return_{w}d_after"] = np.nan
    return result


def classify_reaction(abnormal_1d: float, abnormal_3d: float, abnormal_5d: float) -> str:
    """
    Classify reaction based on abnormal returns.

    Uses fixed thresholds as baseline (can be calibrated later).
    Threshold: ±2% abnormal return over 3 days.
    """
    # Use 3-day abnormal return as primary signal
    primary = abnormal_3d if not np.isnan(abnormal_3d) else abnormal_1d

    if np.isnan(primary):
        return "UNKNOWN"

    if primary > 0.02:
        return "POSITIVE"
    elif primary < -0.02:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("BUILD EVENT → PRICE ML DATASET")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load canonical events
    # ------------------------------------------------------------------
    print("\n[1/6] Loading canonical earnings events...")
    events = pd.read_parquet(EVENTS_INPUT)
    print(f"  Total events: {len(events)}")

    # Filter to events with valid result announcement
    events = events[events["result_announcement_datetime"].notna()].copy()
    events["result_announcement_datetime"] = pd.to_datetime(events["result_announcement_datetime"])
    print(f"  Events with announcement timestamp: {len(events)}")

    # ------------------------------------------------------------------
    # Load price data
    # ------------------------------------------------------------------
    print("\n[2/6] Loading price data...")
    prices = load_all_prices()
    print(f"  Total price rows: {len(prices):,}")
    print(f"  Unique symbols: {prices['symbol'].nunique()}")

    # Get NIFTY 50 symbols
    nifty50 = pd.read_csv(REFERENCE_FILE)["symbol"].tolist()
    prices_nifty = prices[prices["symbol"].isin(nifty50)].copy()
    print(f"  NIFTY 50 price rows: {len(prices_nifty):,}")
    print(f"  NIFTY 50 symbols in prices: {prices_nifty['symbol'].nunique()}")

    # Get benchmark prices (NIFTYBEES)
    bench_prices = prices[prices["symbol"] == BENCHMARK_SYMBOL].copy()
    bench_prices = bench_prices.sort_values("date").reset_index(drop=True)
    print(f"  Benchmark ({BENCHMARK_SYMBOL}) rows: {len(bench_prices)}")
    print(f"  Benchmark date range: {bench_prices['date'].min()} to {bench_prices['date'].max()}")

    # ------------------------------------------------------------------
    # Build trading day infrastructure
    # ------------------------------------------------------------------
    print("\n[3/6] Building trading day calendar...")
    trading_days = get_trading_days(prices)
    date_to_idx, idx_to_date = build_trading_day_lookup(trading_days)
    print(f"  Trading days: {len(trading_days)} ({trading_days[0]} to {trading_days[-1]})")

    # Create price lookup: symbol -> Series indexed by trading day position
    print("\n[4/6] Building price lookup tables...")
    price_lookup = {}
    volume_lookup = {}
    for sym in nifty50:
        sym_prices = prices_nifty[prices_nifty["symbol"] == sym].sort_values("date")
        if len(sym_prices) > 0:
            # Align close prices
            aligned_close = pd.Series(index=range(len(trading_days)), dtype=float)
            # Align volumes
            aligned_vol = pd.Series(index=range(len(trading_days)), dtype=float)
            for _, row in sym_prices.iterrows():
                d = row["date"].date()
                if d in date_to_idx:
                    idx = date_to_idx[d]
                    aligned_close.iloc[idx] = row["close"]
                    aligned_vol.iloc[idx] = row["volume"]
            price_lookup[sym] = aligned_close
            volume_lookup[sym] = aligned_vol
    print(f"  Built price lookup for {len(price_lookup)} symbols")
    print(f"  Built volume lookup for {len(volume_lookup)} symbols")

    # Benchmark aligned series
    bench_aligned = pd.Series(index=range(len(trading_days)), dtype=float)
    for _, row in bench_prices.iterrows():
        d = row["date"].date()
        if d in date_to_idx:
            bench_aligned.iloc[date_to_idx[d]] = row["close"]

    # ------------------------------------------------------------------
    # Process each event
    # ------------------------------------------------------------------
    print("\n[5/6] Computing features and labels for each event...")
    rows = []
    skipped = {
        "no_announcement": 0,
        "no_cutoff": 0,
        "no_reaction_start": 0,
        "no_price_history": 0,
        "insufficient_history": 0,
        "no_benchmark": 0,
    }

    for _, event in events.iterrows():
        event_key = event["event_key"]
        symbol = event["symbol"]
        ann_dt = event["result_announcement_datetime"]

        # Get feature cutoff date
        cutoff_dt = get_feature_cutoff_date(ann_dt, trading_days, date_to_idx)
        if cutoff_dt is None:
            skipped["no_cutoff"] += 1
            continue

        # Get reaction start date
        reaction_dt = get_reaction_start_date(ann_dt, trading_days, date_to_idx)
        if reaction_dt is None:
            skipped["no_reaction_start"] += 1
            continue

        cutoff_idx = date_to_idx[cutoff_dt.date()]
        reaction_idx = date_to_idx[reaction_dt.date()]

        # Get stock price series up to cutoff
        if symbol not in price_lookup:
            skipped["no_price_history"] += 1
            continue

        stock_series = price_lookup[symbol]
        # Only use data up to and including cutoff
        hist_prices = stock_series.iloc[:cutoff_idx + 1].dropna()

        # Check minimum history
        if len(hist_prices) < 2:
            skipped["insufficient_history"] += 1
            continue

        # Benchmark history up to cutoff
        bench_hist = bench_aligned.iloc[:cutoff_idx + 1].dropna()
        if len(bench_hist) < 2:
            skipped["no_benchmark"] += 1
            if skipped["no_benchmark"] <= 5:
                print(f"  DEBUG no_benchmark: {event_key}, cutoff_idx={cutoff_idx}, bench_hist_len={len(bench_hist)}")
            continue

        # Calculate returns for feature windows
        hist_returns = hist_prices.pct_change().dropna()
        bench_hist_returns = bench_hist.pct_change().dropna()

        # Build feature dict
        features = {}

        # Price returns
        features.update(calculate_returns(hist_prices, RETURN_WINDOWS))

        # Volatility
        features.update(calculate_volatility(hist_returns, VOLATILITY_WINDOWS))

        # Volume features (need volume data)
        if symbol in volume_lookup:
            vol_series = volume_lookup[symbol]
            hist_vol = vol_series.iloc[:cutoff_idx + 1].dropna()
            features.update(calculate_volume_features(hist_vol, VOLUME_WINDOWS))
        else:
            for w in VOLUME_WINDOWS:
                features[f"volume_ratio_{w}d"] = np.nan
                features[f"volume_zscore_{w}d"] = np.nan

        # MA distance
        features.update(calculate_ma_distance(hist_prices, MA_WINDOWS))

        # Drawdown
        features.update(calculate_drawdown(hist_prices, DRAWDOWN_WINDOWS))

        # Benchmark-relative returns
        bench_returns_dict = calculate_returns(bench_hist, RETURN_WINDOWS)
        features.update(calculate_relative_returns(features, bench_returns_dict, RETURN_WINDOWS))

        # Calculate labels (post-event)
        # Need forward prices from reaction_idx
        stock_forward = stock_series.iloc[reaction_idx:].dropna()
        bench_forward = bench_aligned.iloc[reaction_idx:].dropna()

        labels = calculate_abnormal_returns(
            stock_forward, bench_forward, LABEL_WINDOWS, 0, trading_days
        )

        # Reaction classification
        reaction_class = classify_reaction(
            labels.get("abnormal_return_1d", np.nan),
            labels.get("abnormal_return_3d", np.nan),
            labels.get("abnormal_return_5d", np.nan),
        )

        # Data quality flags
        quality_flags = []
        for feat_name, min_hist in MIN_HISTORY_FOR_FEATURES.items():
            if feat_name in features and np.isnan(features[feat_name]):
                quality_flags.append(f"INSUFFICIENT_HISTORY_{feat_name}")

        if len(hist_prices) < 60:
            quality_flags.append("LIMITED_HISTORY")
        if len(hist_prices) < 252:
            quality_flags.append("VERY_LIMITED_HISTORY")

        # Session type
        session_type = classify_announcement_session(ann_dt, trading_days)

        # Build output row
        row = {
            "event_key": event_key,
            "symbol": symbol,
            "company_name": event["company_name"],
            "period_ended": event["period_ended"],
            "fiscal_quarter": event["fiscal_quarter"],
            "result_announcement_datetime": ann_dt,
            "announcement_session_type": session_type,
            "feature_cutoff_date": cutoff_dt,
            "reaction_start_date": reaction_dt,
            "reaction_class": reaction_class,
            "quality_flags": "|".join(quality_flags) if quality_flags else "OK",
            "price_history_days": len(hist_prices),
            "benchmark_history_days": len(bench_hist),
            **features,
            **labels,
        }
        rows.append(row)

    print(f"  Processed {len(rows)} events")
    for k, v in skipped.items():
        if v > 0:
            print(f"  Skipped ({k}): {v}")

    # ------------------------------------------------------------------
    # Build output DataFrame
    # ------------------------------------------------------------------
    print("\n[6/6] Building output dataset...")
    df = pd.DataFrame(rows)

    # Ensure column ordering
    meta_cols = [
        "event_key", "symbol", "company_name", "period_ended", "fiscal_quarter",
        "result_announcement_datetime", "announcement_session_type",
        "feature_cutoff_date", "reaction_start_date",
        "reaction_class", "quality_flags",
        "price_history_days", "benchmark_history_days",
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols and not c.endswith("_after")]
    label_cols = [c for c in df.columns if c.endswith("_after")]

    # Order: meta + features (by window) + labels
    ordered_cols = meta_cols + sorted(feature_cols) + sorted(label_cols)
    df = df[ordered_cols]

    # Save
    df.to_parquet(EVENTS_OUTPUT, index=False)
    print(f"  Saved: {EVENTS_OUTPUT} ({len(df)} rows, {len(df.columns)} columns)")

    # Quality report
    quality_report = df[meta_cols].copy()
    quality_report.to_csv(QUALITY_REPORT_OUTPUT, index=False)
    print(f"  Quality report: {QUALITY_REPORT_OUTPUT}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total canonical events (with announcement): {len(events)}")
    print(f"ML rows generated: {len(df)}")
    print(f"Unique symbols: {df['symbol'].nunique()}")
    print(f"Unique periods: {df['period_ended'].nunique()}")

    print("\nReaction class distribution:")
    print(df["reaction_class"].value_counts().to_string())

    print("\nSession type distribution:")
    print(df["announcement_session_type"].value_counts().to_string())

    print("\nFeature availability (non-null %):")
    for col in feature_cols:
        pct = df[col].notna().mean() * 100
        print(f"  {col}: {pct:.1f}%")

    print("\nLabel availability (non-null %):")
    for col in label_cols:
        pct = df[col].notna().mean() * 100
        print(f"  {col}: {pct:.1f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()