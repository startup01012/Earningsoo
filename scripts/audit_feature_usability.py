#!/usr/bin/env python
"""
Feature Usability Audit

Audits every modeled feature in the ML-ready dataset and classifies them as:
- CORE: High usability, low missingness, valid semantics
- SPARSE_BUT_VALID: High missingness but valid when present, economically meaningful
- UNUSABLE: All missing, constant, infinite, target leakage, post-event, or impossible semantics

Outputs: data/processed/feature_usability_audit.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def load_data():
    """Load ML-ready dataset and feature manifest."""
    df = pd.read_parquet('data/processed/earnings_ml_ready.parquet')
    manifest = pd.read_csv('data/processed/earnings_feature_manifest.csv')
    return df, manifest


def get_feature_columns(df, manifest):
    """Get feature columns (non-target, non-identifier, non-audit)."""
    identifier_cols = [
        'event_key', 'symbol', 'company_name', 'period_ended',
        'fiscal_quarter', 'result_announcement_datetime',
        'announcement_session_type', 'feature_cutoff_date',
        'reaction_start_date', 'quality_flags',
        'price_history_days', 'benchmark_history_days'
    ]
    audit_cols = [
        'fundamental_period_end_date', 'fundamental_estimated_available_date',
        'fundamental_pit_status', 'fundamental_lag_days'
    ]
    target_cols = manifest[manifest['is_target'] == True]['feature_name'].tolist()

    feature_cols = [c for c in df.columns if c not in identifier_cols and c not in audit_cols and c not in target_cols]
    return feature_cols, target_cols, identifier_cols, audit_cols


def get_splits(df):
    """Get train/validation/test splits based on period_ended."""
    # Based on checkpoint 51 split methodology
    train_max = pd.Timestamp('2021-12-31')
    val_max = pd.Timestamp('2024-03-31')

    df['period_ended_dt'] = pd.to_datetime(df['period_ended'])

    train_mask = df['period_ended_dt'] <= train_max
    val_mask = (df['period_ended_dt'] > train_max) & (df['period_ended_dt'] <= val_max)
    test_mask = df['period_ended_dt'] > val_max

    return train_mask, val_mask, test_mask


def check_constant(series, threshold=1):
    """Check if feature is constant or near-constant."""
    nunique = series.nunique(dropna=True)
    return nunique <= threshold


def check_near_constant(series, threshold=0.01):
    """Check if feature is near-constant (very low variance relative to mean)."""
    if series.dtype.kind not in 'fc':
        return False
    non_null = series.dropna()
    if len(non_null) < 2:
        return True
    cv = non_null.std() / (abs(non_null.mean()) + 1e-10)
    return cv < threshold


def check_extreme_values(series, z_threshold=10):
    """Check for extreme values using modified z-score."""
    if series.dtype.kind not in 'fc':
        return False
    non_null = series.dropna()
    if len(non_null) < 4:
        return False
    median = non_null.median()
    mad = (non_null - median).abs().median()
    if mad == 0:
        return False
    modified_z = 0.6745 * (non_null - median) / mad
    return (modified_z.abs() > z_threshold).any()


def check_coverage_shift(train_series, val_series, test_series, threshold=0.3):
    """Check if missingness differs substantially between splits."""
    train_miss = train_series.isna().mean()
    val_miss = val_series.isna().mean()
    test_miss = test_series.isna().mean()

    miss_rates = [train_miss, val_miss, test_miss]
    max_diff = max(miss_rates) - min(miss_rates)
    return max_diff > threshold


def check_distribution_shift(train_series, val_series, test_series, alpha=0.01):
    """Check for distribution shift between splits using KS test."""
    if train_series.dtype.kind not in 'fc':
        return False

    train_vals = train_series.dropna().values
    val_vals = val_series.dropna().values
    test_vals = test_series.dropna().values

    if len(train_vals) < 10 or len(val_vals) < 10 or len(test_vals) < 10:
        return False

    # KS test between train and validation
    try:
        _, p_val = stats.ks_2samp(train_vals, val_vals)
        if p_val < alpha:
            return True
    except Exception:
        pass

    # KS test between train and test
    try:
        _, p_val = stats.ks_2samp(train_vals, test_vals)
        if p_val < alpha:
            return True
    except Exception:
        pass

    return False


def classify_feature(row, df, feature_cols, train_mask, val_mask, test_mask, manifest_row=None):
    """Classify a single feature as CORE, SPARSE_BUT_VALID, or UNUSABLE."""
    col = row['feature_name']
    series = df[col]

    # Get basic stats
    overall_missing = series.isna().mean()
    overall_unique = series.nunique(dropna=True)
    overall_var = series.var(skipna=True) if series.dtype.kind in 'fc' else np.nan
    inf_count = np.sum(np.isinf(series.values)) if series.dtype.kind in 'fc' else 0

    # Split-specific stats
    train_series = series[train_mask]
    val_series = series[val_mask]
    test_series = series[test_mask]

    train_missing = train_series.isna().mean()
    val_missing = val_series.isna().mean()
    test_missing = test_series.isna().mean()

    train_unique = train_series.nunique(dropna=True)
    val_unique = val_series.nunique(dropna=True)
    test_unique = test_series.nunique(dropna=True)

    train_var = train_series.var(skipna=True) if train_series.dtype.kind in 'fc' else np.nan
    val_var = val_series.var(skipna=True) if val_series.dtype.kind in 'fc' else np.nan
    test_var = test_series.var(skipna=True) if test_series.dtype.kind in 'fc' else np.nan

    # Flags
    constant_flag = check_constant(series)
    near_constant_flag = check_near_constant(series)
    extreme_value_flag = check_extreme_values(series)
    coverage_shift_flag = check_coverage_shift(train_series, val_series, test_series)
    distribution_shift_flag = check_distribution_shift(train_series, val_series, test_series)

    # Determine usability
    reasons = []

    # UNUSABLE conditions
    if overall_missing >= 1.0:
        reasons.append("ALL_MISSING")
    elif constant_flag:
        reasons.append("CONSTANT")
    elif inf_count > 0:
        reasons.append("INFINITE_VALUES")
    elif col in ['reaction_class', 'abnormal_return_1d', 'abnormal_return_3d', 'abnormal_return_5d',
                 'return_1d_after', 'return_3d_after', 'return_5d_after',
                 'benchmark_return_1d_after', 'benchmark_return_3d_after', 'benchmark_return_5d_after',
                 'reaction_start_date']:
        reasons.append("TARGET_LEAKAGE")
    elif '_after' in col:
        reasons.append("POST_EVENT")

    if reasons:
        return 'UNUSABLE', '; '.join(reasons)

    # SPARSE_BUT_VALID conditions
    if overall_missing > 0.5:
        return 'SPARSE_BUT_VALID', f'HIGH_MISSINGNESS ({overall_missing:.1%})'

    # Check for target leakage patterns in feature names
    # Note: return_* and relative_return_* are PRE-EVENT market features (up to feature_cutoff)
    # The targets are return_*_after and benchmark_return_*_after
    target_patterns = ['reaction_class', 'abnormal_return', 'return_1d_after', 'return_3d_after', 'return_5d_after',
                       'benchmark_return_1d_after', 'benchmark_return_3d_after', 'benchmark_return_5d_after']
    if any(pattern == col or col.startswith(pattern + '_') for pattern in ['reaction_class', 'abnormal_return']):
        if not col.startswith('prior_'):
            return 'UNUSABLE', 'TARGET_LEAKAGE_PATTERN'

    # CORE
    return 'CORE', ''


def main():
    print("Loading data...")
    df, manifest = load_data()

    feature_cols, target_cols, identifier_cols, audit_cols = get_feature_columns(df, manifest)
    print(f"Found {len(feature_cols)} feature columns")

    train_mask, val_mask, test_mask = get_splits(df)
    print(f"Split sizes: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

    # Filter manifest to features only
    feature_manifest = manifest[manifest['is_target'] == False].copy()

    results = []
    for _, mrow in feature_manifest.iterrows():
        col = mrow['feature_name']
        if col not in df.columns:
            print(f"WARNING: Feature {col} in manifest but not in dataset")
            continue

        try:
            usability, reason = classify_feature(mrow, df, feature_cols, train_mask, val_mask, test_mask)

            series = df[col]
            train_series = series[train_mask]
            val_series = series[val_mask]
            test_series = series[test_mask]

            result = {
                'feature_name': col,
                'feature_group': mrow['feature_group'],
                'dtype': str(df[col].dtype),
                'train_missing_pct': round(train_series.isna().mean() * 100, 2),
                'validation_missing_pct': round(val_series.isna().mean() * 100, 2),
                'test_missing_pct': round(test_series.isna().mean() * 100, 2),
                'overall_missing_pct': round(series.isna().mean() * 100, 2),
                'train_unique_count': int(train_series.nunique(dropna=True)),
                'validation_unique_count': int(val_series.nunique(dropna=True)),
                'test_unique_count': int(test_series.nunique(dropna=True)),
                'overall_unique_count': int(series.nunique(dropna=True)),
                'train_variance': round(train_series.var(skipna=True), 6) if train_series.dtype.kind in 'fc' else np.nan,
                'validation_variance': round(val_series.var(skipna=True), 6) if val_series.dtype.kind in 'fc' else np.nan,
                'test_variance': round(test_series.var(skipna=True), 6) if test_series.dtype.kind in 'fc' else np.nan,
                'infinite_count': int(np.sum(np.isinf(series.values))) if series.dtype.kind in 'fc' else 0,
                'constant_flag': check_constant(series),
                'near_constant_flag': check_near_constant(series),
                'extreme_value_flag': check_extreme_values(series),
                'coverage_shift_flag': check_coverage_shift(train_series, val_series, test_series),
                'distribution_shift_flag': check_distribution_shift(train_series, val_series, test_series),
                'usability_status': usability,
                'usability_reason': reason
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing {col}: {e}")
            results.append({
                'feature_name': col,
                'feature_group': mrow['feature_group'],
                'dtype': str(df[col].dtype) if col in df.columns else 'MISSING',
                'train_missing_pct': np.nan,
                'validation_missing_pct': np.nan,
                'test_missing_pct': np.nan,
                'overall_missing_pct': np.nan,
                'train_unique_count': np.nan,
                'validation_unique_count': np.nan,
                'test_unique_count': np.nan,
                'overall_unique_count': np.nan,
                'train_variance': np.nan,
                'validation_variance': np.nan,
                'test_variance': np.nan,
                'infinite_count': np.nan,
                'constant_flag': np.nan,
                'near_constant_flag': np.nan,
                'extreme_value_flag': np.nan,
                'coverage_shift_flag': np.nan,
                'distribution_shift_flag': np.nan,
                'usability_status': 'ERROR',
                'usability_reason': str(e)
            })

    results_df = pd.DataFrame(results)

    # Save
    output_path = 'data/processed/feature_usability_audit.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Summary
    print("\n=== USABILITY SUMMARY ===")
    print(results_df['usability_status'].value_counts())
    print("\nBy feature group:")
    print(results_df.groupby(['feature_group', 'usability_status']).size().unstack(fill_value=0))

    # Show UNUSABLE features
    unusable = results_df[results_df['usability_status'] == 'UNUSABLE']
    if len(unusable) > 0:
        print("\n=== UNUSABLE FEATURES ===")
        for _, r in unusable.iterrows():
            print(f"  {r['feature_name']} ({r['feature_group']}): {r['usability_reason']}")

    # Show SPARSE_BUT_VALID features
    sparse = results_df[results_df['usability_status'] == 'SPARSE_BUT_VALID']
    if len(sparse) > 0:
        print("\n=== SPARSE_BUT_VALID FEATURES ===")
        for _, r in sparse.iterrows():
            print(f"  {r['feature_name']} ({r['feature_group']}): {r['overall_missing_pct']:.1f}% missing - {r['usability_reason']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())