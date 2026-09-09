"""
Build Expanded Prior-Earnings Features (Leakage-Safe)

For each earnings event, computes rolling statistics from PRIOR events only.
Never uses the current event's outcome or any future event data.

Output: data/processed/earnings_event_prior_earnings_features.parquet
"""

import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

# Full canonical events (not just ML subset)
EVENTS_INPUT = Path("data/processed/earnings_events.parquet")

# Current ML dataset with all features so far
ML_INPUT = Path("data/processed/earnings_event_valuation_features.parquet")

OUTPUT_FILE = Path("data/processed/earnings_event_prior_earnings_features.parquet")
QUALITY_REPORT = Path("data/processed/earnings_event_prior_earnings_features_quality_report.csv")

for p in [OUTPUT_FILE, QUALITY_REPORT]:
    p.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================

def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safe division with zero/NaN handling."""
    result = num / den
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def compute_prior_earnings_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute leakage-safe prior-earnings features for each event.
    
    For event i of a symbol, only uses events 0 to i-1.
    """
    # Ensure sorted by symbol and period_ended
    events = events.sort_values(["symbol", "period_ended"]).reset_index(drop=True)
    
    # We need the reaction labels to compute prior features
    # Check if they exist
    has_labels = all(c in events.columns for c in ["abnormal_return_1d", "abnormal_return_3d", "abnormal_return_5d", "reaction_class"])
    
    if not has_labels:
        print("  WARNING: Reaction labels not in events table. Computing from ML dataset...")
        # Load ML dataset to get labels
        ml_data = pd.read_parquet(ML_INPUT)
        label_cols = ["event_key", "abnormal_return_1d", "abnormal_return_3d", "abnormal_return_5d", "reaction_class"]
        events = events.merge(ml_data[label_cols], on="event_key", how="left")
        has_labels = True
    
    # Convert reaction_class to numeric for calculations
    reaction_map = {"NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1, "UNKNOWN": np.nan}
    events["reaction_numeric"] = events["reaction_class"].map(reaction_map)
    
    # Group by symbol for rolling computations
    grouped = events.groupby("symbol")
    
    # Initialize feature columns
    feature_cols = [
        "prior_earnings_count",
        "prior_abn_ret_1d_mean_2", "prior_abn_ret_1d_mean_4", "prior_abn_ret_1d_mean_8",
        "prior_abn_ret_3d_mean_2", "prior_abn_ret_3d_mean_4", "prior_abn_ret_3d_mean_8",
        "prior_abn_ret_5d_mean_2", "prior_abn_ret_5d_mean_4", "prior_abn_ret_5d_mean_8",
        "prior_abn_ret_1d_median_4", "prior_abn_ret_1d_median_8",
        "prior_abn_ret_3d_median_4", "prior_abn_ret_3d_median_8",
        "prior_abn_ret_1d_std_4", "prior_abn_ret_1d_std_8",
        "prior_abn_ret_3d_std_4", "prior_abn_ret_3d_std_8",
        "prior_positive_rate_4", "prior_positive_rate_8",
        "prior_negative_rate_4", "prior_negative_rate_8",
        "prior_neutral_rate_4", "prior_neutral_rate_8",
        "prior_reaction_streak",  # Consecutive same reaction
        "prior_reaction_trend_4", "prior_reaction_trend_8",  # Linear slope
        "prior_avg_reaction_magnitude_4", "prior_avg_reaction_magnitude_8",
        "prior_days_since_last",
        "prior_last_reaction_class",
        "prior_last_abn_ret_1d", "prior_last_abn_ret_3d", "prior_last_abn_ret_5d",
    ]
    
    all_features = []
    
    for symbol, group in grouped:
        group = group.reset_index(drop=True)
        n = len(group)
        
        for i in range(n):
            row = group.iloc[i]
            event_key = row["event_key"]
            
            # Get prior events (strictly before current)
            prior = group.iloc[:i]
            prior_count = len(prior)
            
            features = {"event_key": event_key, "prior_earnings_count": prior_count}
            
            if prior_count == 0:
                # No prior earnings - all NaN
                for col in feature_cols:
                    if col not in features:
                        features[col] = np.nan
                all_features.append(features)
                continue
            
            # Last prior event
            last = prior.iloc[-1]
            features["prior_last_reaction_class"] = last["reaction_class"]
            features["prior_last_abn_ret_1d"] = last.get("abnormal_return_1d", np.nan)
            features["prior_last_abn_ret_3d"] = last.get("abnormal_return_3d", np.nan)
            features["prior_last_abn_ret_5d"] = last.get("abnormal_return_5d", np.nan)
            features["prior_days_since_last"] = (row["period_ended"] - last["period_ended"]).days
            
            # Rolling windows
            for window in [2, 4, 8]:
                if prior_count >= window:
                    recent = prior.iloc[-window:]
                    
                    # Mean abnormal returns
                    for w in [1, 3, 5]:
                        col = f"abnormal_return_{w}d"
                        if col in recent.columns:
                            features[f"prior_abn_ret_{w}d_mean_{window}"] = recent[col].mean()
                            if window >= 4:
                                features[f"prior_abn_ret_{w}d_median_{window}"] = recent[col].median()
                            if window >= 4:
                                features[f"prior_abn_ret_{w}d_std_{window}"] = recent[col].std()
                    
                    # Reaction rates
                    if window >= 4:
                        react = recent["reaction_class"]
                        features[f"prior_positive_rate_{window}"] = (react == "POSITIVE").mean()
                        features[f"prior_negative_rate_{window}"] = (react == "NEGATIVE").mean()
                        features[f"prior_neutral_rate_{window}"] = (react == "NEUTRAL").mean()
                    
                    # Average reaction magnitude (absolute abnormal return)
                    for w in [1, 3, 5]:
                        col = f"abnormal_return_{w}d"
                        if col in recent.columns:
                            features[f"prior_avg_reaction_magnitude_{window}"] = recent[col].abs().mean()
                    
                    # Reaction trend (linear slope of reaction_numeric)
                    if window >= 4:
                        react_num = recent["reaction_numeric"].dropna()
                        if len(react_num) >= 2:
                            x = np.arange(len(react_num))
                            slope = np.polyfit(x, react_num, 1)[0]
                            features[f"prior_reaction_trend_{window}"] = slope
                        else:
                            features[f"prior_reaction_trend_{window}"] = np.nan
                
                else:
                    # Not enough data for this window
                    for w in [1, 3, 5]:
                        features[f"prior_abn_ret_{w}d_mean_{window}"] = np.nan
                        if window >= 4:
                            features[f"prior_abn_ret_{w}d_median_{window}"] = np.nan
                            features[f"prior_abn_ret_{w}d_std_{window}"] = np.nan
                    if window >= 4:
                        features[f"prior_positive_rate_{window}"] = np.nan
                        features[f"prior_negative_rate_{window}"] = np.nan
                        features[f"prior_neutral_rate_{window}"] = np.nan
                        features[f"prior_avg_reaction_magnitude_{window}"] = np.nan
                        features[f"prior_reaction_trend_{window}"] = np.nan
            
            # Reaction streak (consecutive same reaction)
            streak = 1
            last_reaction = last["reaction_class"]
            for j in range(len(prior) - 2, -1, -1):
                if prior.iloc[j]["reaction_class"] == last_reaction:
                    streak += 1
                else:
                    break
            features["prior_reaction_streak"] = streak
            
            all_features.append(features)
    
    return pd.DataFrame(all_features)


def build_prior_earnings_features():
    """Main pipeline."""
    print("=" * 70)
    print("BUILD EXPANDED PRIOR-EARNINGS FEATURES")
    print("=" * 70)
    
    # Load full canonical events
    print("\n[1/4] Loading canonical earnings events...")
    events = pd.read_parquet(EVENTS_INPUT)
    print(f"  Total events: {len(events)}")
    print(f"  Symbols: {events['symbol'].nunique()}")
    
    # Load ML dataset to get reaction labels
    print("\n[2/4] Loading ML dataset for labels...")
    ml_data = pd.read_parquet(ML_INPUT)
    label_cols = ["event_key", "abnormal_return_1d", "abnormal_return_3d", "abnormal_return_5d", "reaction_class"]
    events = events.merge(ml_data[label_cols], on="event_key", how="left")
    print(f"  Events with labels: {events['reaction_class'].notna().sum()}")
    
    # Compute prior features
    print("\n[3/4] Computing prior-earnings features...")
    prior_features = compute_prior_earnings_features(events)
    print(f"  Generated features for {len(prior_features)} events")
    
    # Merge with ML dataset
    print("\n[4/4] Merging with ML dataset...")
    ml_full = pd.read_parquet(ML_INPUT)
    output = ml_full.merge(prior_features, on="event_key", how="left")
    
    # Reorder columns
    prior_cols = [c for c in output.columns if c.startswith("prior_")]
    print(f"  Prior-earnings columns added: {len(prior_cols)}")
    
    # Save
    output.to_parquet(OUTPUT_FILE, index=False)
    print(f"  Saved: {OUTPUT_FILE} ({len(output)} rows, {len(output.columns)} columns)")
    
    # Quality report
    quality_cols = ["event_key", "symbol", "period_ended"] + prior_cols
    output[quality_cols].to_csv(QUALITY_REPORT, index=False)
    print(f"  Quality report: {QUALITY_REPORT}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PRIOR-EARNINGS FEATURE SUMMARY")
    print("=" * 70)
    
    for col in sorted(prior_cols):
        pct = output[col].notna().mean() * 100
        print(f"  {col}: {pct:.1f}% non-null")
    
    print("\nDone!")
    return output


if __name__ == "__main__":
    build_prior_earnings_features()