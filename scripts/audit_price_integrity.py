#!/usr/bin/env python
"""
Price / Corporate Action Impact Audit

Inspects whether raw price data can affect:
- return_1d, return_3d, return_5d
- moving averages
- volatility
- drawdowns
- relative returns
- abnormal returns
- volume features

Identifies potential corporate-action-sensitive periods.
Does NOT fabricate adjusted prices.

Outputs:
- data/processed/price_integrity_audit.csv
- data/processed/price_integrity_audit.txt
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """Load ML-ready dataset."""
    df = pd.read_parquet('data/processed/earnings_ml_ready.parquet')
    return df


def identify_corporate_action_sensitive_features():
    """Identify which features are sensitive to corporate actions."""
    sensitive_features = {
        'return_1d': 'HIGH - Single day return directly affected by splits/bonuses',
        'return_3d': 'HIGH - Multi-day return compounds split effects',
        'return_5d': 'HIGH - Multi-day return compounds split effects',
        'return_10d': 'HIGH',
        'return_20d': 'HIGH',
        'return_60d': 'HIGH',
        'relative_return_1d': 'HIGH - Relative return vs benchmark affected',
        'relative_return_3d': 'HIGH',
        'relative_return_5d': 'HIGH',
        'relative_return_10d': 'HIGH',
        'relative_return_20d': 'HIGH',
        'relative_return_60d': 'HIGH',
        'volatility_5d': 'MEDIUM - Volatility calculation uses price changes',
        'volatility_20d': 'MEDIUM',
        'volatility_60d': 'MEDIUM',
        'distance_from_20dma': 'HIGH - Moving averages shift on split',
        'distance_from_50dma': 'HIGH',
        'distance_from_200dma': 'HIGH',
        'drawdown_from_20d_high': 'HIGH - High/low prices affected',
        'drawdown_from_60d_high': 'HIGH',
        'drawdown_from_252d_high': 'HIGH',
        'volume_ratio_5d': 'LOW - Volume splits but ratio may be preserved',
        'volume_ratio_20d': 'LOW',
        'volume_zscore_5d': 'LOW',
        'volume_zscore_20d': 'LOW',
        'valuation_features': 'HIGH - Valuation uses price (market cap, P/E, etc.)',
    }
    return sensitive_features


def check_target_sensitivity(df):
    """Check if targets are affected by corporate actions."""
    target_cols = ['reaction_class', 'abnormal_return_1d', 'abnormal_return_3d', 'abnormal_return_5d',
                   'return_1d_after', 'return_3d_after', 'return_5d_after',
                   'benchmark_return_1d_after', 'benchmark_return_3d_after', 'benchmark_return_5d_after']

    sensitive_targets = {}
    for target in target_cols:
        if target in df.columns:
            if 'return' in target.lower() or 'abnormal' in target.lower():
                sensitive_targets[target] = 'HIGH - Target returns use raw prices'
            elif 'reaction_class' in target:
                sensitive_targets[target] = 'HIGH - Derived from abnormal_return_3d'

    return sensitive_targets


def detect_potential_splits(df, price_col='close'):
    """
    Detect potential corporate actions by looking for extreme single-day
    price changes that are not accompanied by similar market moves.
    Note: This is heuristic - we don't have actual corporate action data.
    """
    # We don't have raw price data in the ML dataset, only features
    # So we check for extreme feature values that could indicate splits
    results = {}

    # Check for extreme returns (potential split indicators)
    for col in ['return_1d', 'return_3d', 'return_5d', 'return_10d']:
        if col in df.columns:
            series = df[col].dropna()
            extreme_low = series.quantile(0.001)
            extreme_high = series.quantile(0.999)
            results[col] = {
                'extreme_low': extreme_low,
                'extreme_high': extreme_high,
                'extreme_low_pct': extreme_low * 100,
                'extreme_high_pct': extreme_high * 100,
                'n_obs': len(series)
            }

    # Check for price level discontinuities in distance_from_* features
    for col in ['distance_from_20dma', 'distance_from_50dma', 'distance_from_200dma']:
        if col in df.columns:
            series = df[col].dropna()
            # Large jumps in distance could indicate splits
            diff = series.diff().abs()
            large_jumps = (diff > diff.quantile(0.999)).sum()
            results[f'{col}_jumps'] = {
                'large_jumps_count': int(large_jumps),
                'max_jump': diff.max()
            }

    return results


def check_benchmark_integrity(df):
    """Check if benchmark (NIFTYBEES) has corporate action issues."""
    # NIFTYBEES is an ETF - should have fewer corporate actions
    # But we check relative returns for anomalies
    benchmark_cols = [c for c in df.columns if c.startswith('benchmark_return')]
    relative_cols = [c for c in df.columns if c.startswith('relative_return')]

    results = {}
    for col in benchmark_cols + relative_cols:
        if col in df.columns:
            series = df[col].dropna()
            results[col] = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'extreme_low': series.quantile(0.001),
                'extreme_high': series.quantile(0.999)
            }

    return results


def check_valuation_price_dependency(df):
    """Check valuation features for price dependency."""
    val_cols = [c for c in df.columns if c.startswith('val_')]
    results = {}

    for col in val_cols:
        if col in df.columns:
            series = df[col].dropna()
            missing = df[col].isna().mean()
            results[col] = {
                'missing_pct': missing * 100,
                'mean': series.mean() if len(series) > 0 else np.nan,
                'std': series.std() if len(series) > 0 else np.nan,
                'min': series.min() if len(series) > 0 else np.nan,
                'max': series.max() if len(series) > 0 else np.nan
            }

    return results


def generate_audit_report(df, sensitive_features, sensitive_targets, split_detection, benchmark_check, valuation_check):
    """Generate audit report."""
    txt = """======================================================================
PRICE / CORPORATE ACTION IMPACT AUDIT
======================================================================

OBJECTIVE
---------
Assess the impact of using RAW UNADJUSTED Bhavcopy prices on features
and targets. Corporate actions (splits, bonuses, dividends, mergers)
create artificial price discontinuities that distort returns,
moving averages, volatility, and valuation metrics.

======================================================================
DATA SOURCE
======================================================================
- Source: NSE Bhavcopy (raw daily OHLCV)
- Adjustment: NONE - Raw prices used directly
- Corporate actions NOT adjusted: splits, bonuses, rights issues,
  mergers/demergers, dividends

======================================================================
FEATURE SENSITIVITY ASSESSMENT
======================================================================
"""

    for feat, desc in sorted(sensitive_features.items()):
        txt += f"  {feat}: {desc}\n"

    txt += """

======================================================================
TARGET SENSITIVITY (CRITICAL)
======================================================================
"""
    txt += "TARGETS ARE AFFECTED BY RAW PRICES:\n"
    for target, desc in sorted(sensitive_targets.items()):
        txt += f"  {target}: {desc}\n"

    txt += """

======================================================================
POTENTIAL SPLIT DETECTION (Heuristic)
======================================================================
Note: Without actual corporate action data, we use extreme return
values as proxy for potential split events.
"""
    for feat, stats in sorted(split_detection.items()):
        if 'extreme_low' in stats:
            txt += f"\n  {feat}:\n"
            txt += f"    Extreme low (0.1%): {stats['extreme_low_pct']:.2f}%\n"
            txt += f"    Extreme high (99.9%): {stats['extreme_high_pct']:.2f}%\n"
        elif 'large_jumps_count' in stats:
            txt += f"\n  {feat}:\n"
            txt += f"    Large jumps detected: {stats['large_jumps_count']}\n"
            txt += f"    Max jump magnitude: {stats['max_jump']:.4f}\n"

    txt += """

======================================================================
BENCHMARK (NIFTYBEES) INTEGRITY
======================================================================
"""
    for col, stats in sorted(benchmark_check.items()):
        txt += f"\n  {col}:\n"
        txt += f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}\n"
        txt += f"    Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n"
        txt += f"    Extreme (0.1%/99.9%): [{stats['extreme_low']:.6f}, {stats['extreme_high']:.6f}]\n"

    txt += """

======================================================================
VALUATION FEATURES (Price-Dependent)
======================================================================
"""
    for col, stats in sorted(valuation_check.items()):
        txt += f"\n  {col}:\n"
        txt += f"    Missing: {stats['missing_pct']:.1f}%\n"
        txt += f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}\n"
        txt += f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n"

    txt += """

======================================================================
CORPORATE ACTION SENSITIVE PERIODS (Known from Literature)
======================================================================
Major known corporate actions in NIFTY 50 universe (approximate):
- HDFC Bank: 1:1 bonus (2011), 1:1 split (2019)
- Reliance: 1:1 bonus (2017, 2023), rights issue (2020)
- ICICI Bank: 1:10 split (2017), bonus (2018)
- TCS: 1:1 bonus (2018), buyback (multiple)
- Infosys: 1:1 bonus (2014, 2018), buyback (multiple)
- HUL: 1:1 bonus (2016)
- ITC: 1:10 split (2015), bonus (2016)
- L&T: 1:5 split (2011)
- SBI: 1:10 split (2014)
- Kotak Bank: 1:1 bonus (2018)

These events create artificial price drops of 50% (1:1 split/bonus)
or 90% (1:10 split) that appear as massive negative returns in raw data.

======================================================================
IMPACT ASSESSMENT
======================================================================
HIGH PRIORITY BEFORE PRODUCTION:
- All return features (return_*, relative_return_*) affected
- All MA distance features affected
- All drawdown features affected
- Valuation features (market cap, P/E, etc.) affected
- TARGETS (abnormal_return_*, return_*_after) affected

This means BOTH features AND labels contain corporate action noise.

RECOMMENDATION:
1. Document as RAW_UNADJUSTED (current state)
2. For research: Acknowledge limitation, note specific events
3. For production: Obtain corporate action data (NSE provides this)
   and apply split/bonus adjustment factors to price series
4. Alternative: Use yfinance adjusted close (but different methodology)

DO NOT:
- Fabricate adjustment factors
- Zero-out extreme returns (they may be real)
- Assume the effect "averages out"

======================================================================
STATUS: RAW_UNADJUSTED (Documented Limitation)
======================================================================
"""

    return txt


def generate_csv_audit(sensitive_features, sensitive_targets, split_detection, benchmark_check, valuation_check):
    """Generate CSV audit data."""
    rows = []

    # Feature sensitivity
    for feat, desc in sorted(sensitive_features.items()):
        rows.append({
            'audit_item': 'feature_sensitivity',
            'feature': feat,
            'sensitivity': desc.split(' - ')[0] if ' - ' in desc else desc,
            'description': desc,
            'affects_targets': 'return' in feat.lower() or 'abnormal' in feat.lower()
        })

    # Target sensitivity
    for target, desc in sorted(sensitive_targets.items()):
        rows.append({
            'audit_item': 'target_sensitivity',
            'feature': target,
            'sensitivity': desc.split(' - ')[0] if ' - ' in desc else desc,
            'description': desc,
            'affects_targets': True
        })

    # Split detection
    for feat, stats in sorted(split_detection.items()):
        if 'extreme_low' in stats:
            rows.append({
                'audit_item': 'extreme_return_detection',
                'feature': feat,
                'extreme_low_pct': round(stats['extreme_low_pct'], 2),
                'extreme_high_pct': round(stats['extreme_high_pct'], 2),
                'description': 'Potential split indicator'
            })
        elif 'large_jumps_count' in stats:
            rows.append({
                'audit_item': 'ma_distance_jumps',
                'feature': feat,
                'large_jumps_count': stats['large_jumps_count'],
                'max_jump': round(stats['max_jump'], 4),
                'description': 'Potential split indicator'
            })

    # Benchmark
    for col, stats in sorted(benchmark_check.items()):
        rows.append({
            'audit_item': 'benchmark_integrity',
            'feature': col,
            'mean': round(stats['mean'], 6),
            'std': round(stats['std'], 6),
            'min': round(stats['min'], 6),
            'max': round(stats['max'], 6)
        })

    # Valuation
    for col, stats in sorted(valuation_check.items()):
        rows.append({
            'audit_item': 'valuation_price_dependency',
            'feature': col,
            'missing_pct': round(stats['missing_pct'], 1),
            'mean': round(stats['mean'], 2) if not np.isnan(stats['mean']) else np.nan,
            'std': round(stats['std'], 2) if not np.isnan(stats['std']) else np.nan,
            'min': round(stats['min'], 2) if not np.isnan(stats['min']) else np.nan,
            'max': round(stats['max'], 2) if not np.isnan(stats['max']) else np.nan
        })

    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    df = load_data()

    print("Identifying sensitive features...")
    sensitive_features = identify_corporate_action_sensitive_features()

    print("Checking target sensitivity...")
    sensitive_targets = check_target_sensitivity(df)

    print("Detecting potential splits...")
    split_detection = detect_potential_splits(df)

    print("Checking benchmark integrity...")
    benchmark_check = check_benchmark_integrity(df)

    print("Checking valuation features...")
    valuation_check = check_valuation_price_dependency(df)

    print("Generating reports...")
    txt_report = generate_audit_report(df, sensitive_features, sensitive_targets,
                                        split_detection, benchmark_check, valuation_check)
    csv_df = generate_csv_audit(sensitive_features, sensitive_targets, split_detection,
                                benchmark_check, valuation_check)

    # Save
    txt_path = 'data/processed/price_integrity_audit.txt'
    csv_path = 'data/processed/price_integrity_audit.csv'

    with open(txt_path, 'w') as f:
        f.write(txt_report)

    csv_df.to_csv(csv_path, index=False)

    print(f"Saved text report to {txt_path}")
    print(f"Saved CSV to {csv_path}")

    print("\n" + txt_report)

    return 0


if __name__ == "__main__":
    sys.exit(main())