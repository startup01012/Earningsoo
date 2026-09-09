#!/usr/bin/env python
"""
Target Quality Audit

Audits all target variables:
- reaction_class (categorical)
- abnormal_return_1d/3d/5d (continuous)
- return_1d/3d/5d_after (continuous)
- benchmark_return_1d/3d/5d_after (continuous)

Checks:
- Class distribution
- Distribution by year/quarter/symbol
- Train/val/test distribution
- Extreme returns
- Missing/infinite labels
- Duplicate target rows
- Impossible target values
- Label construction consistency

Outputs: data/processed/target_quality_audit.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path


def load_data():
    """Load ML-ready dataset and feature manifest."""
    df = pd.read_parquet('data/processed/earnings_ml_ready.parquet')
    manifest = pd.read_csv('data/processed/earnings_feature_manifest.csv')
    return df, manifest


def get_splits(df):
    """Get train/validation/test splits based on period_ended."""
    train_max = pd.Timestamp('2021-12-31')
    val_max = pd.Timestamp('2024-03-31')

    df['period_ended_dt'] = pd.to_datetime(df['period_ended'])

    train_mask = df['period_ended_dt'] <= train_max
    val_mask = (df['period_ended_dt'] > train_max) & (df['period_ended_dt'] <= val_max)
    test_mask = df['period_ended_dt'] > val_max

    return train_mask, val_mask, test_mask


def audit_reaction_class(df, train_mask, val_mask, test_mask):
    """Audit reaction_class target."""
    results = []

    # Overall distribution
    overall_dist = df['reaction_class'].value_counts()
    overall_pct = df['reaction_class'].value_counts(normalize=True)

    for cls in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']:
        results.append({
            'target': 'reaction_class',
            'class': cls,
            'split': 'OVERALL',
            'count': int(overall_dist.get(cls, 0)),
            'pct': round(overall_pct.get(cls, 0) * 100, 2),
            'check': 'class_distribution'
        })

    # By year
    df['year'] = pd.to_datetime(df['period_ended']).dt.year
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        year_dist = year_df['reaction_class'].value_counts(normalize=True)
        for cls in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']:
            results.append({
                'target': 'reaction_class',
                'class': cls,
                'split': f'YEAR_{year}',
                'count': int(year_df['reaction_class'].value_counts().get(cls, 0)),
                'pct': round(year_dist.get(cls, 0) * 100, 2),
                'check': 'yearly_distribution'
            })

    # By quarter
    df['quarter'] = pd.to_datetime(df['period_ended']).dt.quarter
    for q in sorted(df['quarter'].unique()):
        q_df = df[df['quarter'] == q]
        q_dist = q_df['reaction_class'].value_counts(normalize=True)
        for cls in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']:
            results.append({
                'target': 'reaction_class',
                'class': cls,
                'split': f'QUARTER_Q{q}',
                'count': int(q_df['reaction_class'].value_counts().get(cls, 0)),
                'pct': round(q_dist.get(cls, 0) * 100, 2),
                'check': 'quarterly_distribution'
            })

    # By symbol (top/bottom)
    symbol_dist = df.groupby('symbol')['reaction_class'].value_counts(normalize=True).unstack(fill_value=0)
    for symbol in symbol_dist.index:
        for cls in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']:
            if cls in symbol_dist.columns:
                results.append({
                    'target': 'reaction_class',
                    'class': cls,
                    'split': f'SYMBOL_{symbol}',
                    'count': int((df['symbol'] == symbol).sum() * symbol_dist.loc[symbol, cls]),
                    'pct': round(symbol_dist.loc[symbol, cls] * 100, 2),
                    'check': 'symbol_distribution'
                })

    # Train/Val/Test distribution
    for split_name, mask in [('TRAIN', train_mask), ('VAL', val_mask), ('TEST', test_mask)]:
        split_df = df[mask]
        split_dist = split_df['reaction_class'].value_counts(normalize=True)
        for cls in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']:
            results.append({
                'target': 'reaction_class',
                'class': cls,
                'split': split_name,
                'count': int(split_df['reaction_class'].value_counts().get(cls, 0)),
                'pct': round(split_dist.get(cls, 0) * 100, 2),
                'check': 'split_distribution'
            })

    # Missing check
    missing = df['reaction_class'].isna().sum()
    results.append({
        'target': 'reaction_class',
        'class': 'MISSING',
        'split': 'OVERALL',
        'count': int(missing),
        'pct': round(missing / len(df) * 100, 2),
        'check': 'missing_check'
    })

    # Infinite check (not applicable for categorical)
    results.append({
        'target': 'reaction_class',
        'class': 'INFINITE',
        'split': 'OVERALL',
        'count': 0,
        'pct': 0.0,
        'check': 'infinite_check'
    })

    return results


def audit_continuous_targets(df, train_mask, val_mask, test_mask, target_cols):
    """Audit continuous targets."""
    results = []

    for target in target_cols:
        series = df[target]

        # Overall stats
        results.append({
            'target': target,
            'class': 'ALL',
            'split': 'OVERALL',
            'count': int(series.notna().sum()),
            'pct': round(series.notna().mean() * 100, 2),
            'mean': round(series.mean(), 6),
            'std': round(series.std(), 6),
            'min': round(series.min(), 6),
            'max': round(series.max(), 6),
            'median': round(series.median(), 6),
            'q1': round(series.quantile(0.25), 6),
            'q3': round(series.quantile(0.75), 6),
            'missing_count': int(series.isna().sum()),
            'infinite_count': int(np.sum(np.isinf(series.values))),
            'check': 'overall_stats'
        })

        # By year
        df['year'] = pd.to_datetime(df['period_ended']).dt.year
        for year in sorted(df['year'].unique()):
            year_series = df.loc[df['year'] == year, target]
            results.append({
                'target': target,
                'class': 'ALL',
                'split': f'YEAR_{year}',
                'count': int(year_series.notna().sum()),
                'pct': round(year_series.notna().mean() * 100, 2),
                'mean': round(year_series.mean(), 6),
                'std': round(year_series.std(), 6),
                'min': round(year_series.min(), 6),
                'max': round(year_series.max(), 6),
                'median': round(year_series.median(), 6),
                'q1': round(year_series.quantile(0.25), 6),
                'q3': round(year_series.quantile(0.75), 6),
                'missing_count': int(year_series.isna().sum()),
                'infinite_count': int(np.sum(np.isinf(year_series.values))),
                'check': 'yearly_stats'
            })

        # Train/Val/Test stats
        for split_name, mask in [('TRAIN', train_mask), ('VAL', val_mask), ('TEST', test_mask)]:
            split_series = series[mask]
            results.append({
                'target': target,
                'class': 'ALL',
                'split': split_name,
                'count': int(split_series.notna().sum()),
                'pct': round(split_series.notna().mean() * 100, 2),
                'mean': round(split_series.mean(), 6),
                'std': round(split_series.std(), 6),
                'min': round(split_series.min(), 6),
                'max': round(split_series.max(), 6),
                'median': round(split_series.median(), 6),
                'q1': round(split_series.quantile(0.25), 6),
                'q3': round(split_series.quantile(0.75), 6),
                'missing_count': int(split_series.isna().sum()),
                'infinite_count': int(np.sum(np.isinf(split_series.values))),
                'check': 'split_stats'
            })

    return results


def check_label_consistency(df):
    """Check if reaction_class is consistently derived from abnormal_return_3d."""
    # The threshold is ±2% (0.02)
    threshold = 0.02

    results = []

    # Check consistency
    def expected_class(abn_ret):
        if pd.isna(abn_ret):
            return np.nan
        if abn_ret > threshold:
            return 'POSITIVE'
        elif abn_ret < -threshold:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'

    df['expected_class'] = df['abnormal_return_3d'].apply(expected_class)
    consistent = (df['reaction_class'] == df['expected_class']).sum()
    inconsistent = (df['reaction_class'] != df['expected_class']).sum()
    nan_both = df['reaction_class'].isna() & df['expected_class'].isna()

    results.append({
        'target': 'reaction_class_consistency',
        'class': 'CONSISTENT',
        'split': 'OVERALL',
        'count': int(consistent),
        'pct': round(consistent / len(df) * 100, 2),
        'check': 'label_consistency'
    })
    results.append({
        'target': 'reaction_class_consistency',
        'class': 'INCONSISTENT',
        'split': 'OVERALL',
        'count': int(inconsistent),
        'pct': round(inconsistent / len(df) * 100, 2),
        'check': 'label_consistency'
    })

    # Show inconsistent cases
    if inconsistent > 0:
        inconsistent_df = df[df['reaction_class'] != df['expected_class']]
        for _, row in inconsistent_df.iterrows():
            results.append({
                'target': 'reaction_class_consistency',
                'class': 'INCONSISTENT_DETAIL',
                'split': f"EVENT_{row['event_key']}",
                'count': 1,
                'pct': 0,
                'actual_class': row['reaction_class'],
                'expected_class': row['expected_class'],
                'abnormal_return_3d': row['abnormal_return_3d'],
                'check': 'label_consistency_detail'
            })

    # Check threshold boundary cases
    near_threshold = df[
        (df['abnormal_return_3d'].abs() > threshold * 0.9) &
        (df['abnormal_return_3d'].abs() < threshold * 1.1)
    ]
    results.append({
        'target': 'reaction_class_threshold',
        'class': 'NEAR_THRESHOLD',
        'split': 'OVERALL',
        'count': len(near_threshold),
        'pct': round(len(near_threshold) / len(df) * 100, 2),
        'check': 'threshold_boundary'
    })

    return results


def check_split_boundary_overlap(df, train_mask, val_mask, test_mask):
    """Check if label windows cross split boundaries."""
    # Targets are computed 1/3/5 days AFTER announcement
    # Check if any event's reaction window crosses into another split's training period

    results = []

    df['announcement_dt'] = pd.to_datetime(df['result_announcement_datetime'])
    df['reaction_5d_end'] = df['announcement_dt'] + pd.Timedelta(days=7)  # ~5 trading days

    # For each split, check if reaction window extends into another split's period
    split_dates = {
        'TRAIN_END': pd.Timestamp('2021-12-31'),
        'VAL_END': pd.Timestamp('2024-03-31'),
    }

    for split_name, end_date in split_dates.items():
        # Events in this split
        if split_name == 'TRAIN_END':
            split_mask = train_mask
        else:
            split_mask = val_mask

        split_df = df[split_mask]
        # Check if any reaction_5d_end goes beyond the split boundary
        crosses = split_df[split_df['reaction_5d_end'] > end_date]
        results.append({
            'target': 'split_boundary',
            'class': f'CROSSES_{split_name}',
            'split': 'OVERALL',
            'count': len(crosses),
            'pct': round(len(crosses) / len(split_df) * 100, 2),
            'check': 'boundary_overlap'
        })

    return results


def check_duplicate_targets(df):
    """Check for duplicate target rows."""
    target_cols = ['reaction_class', 'abnormal_return_1d', 'abnormal_return_3d', 'abnormal_return_5d',
                   'return_1d_after', 'return_3d_after', 'return_5d_after',
                   'benchmark_return_1d_after', 'benchmark_return_3d_after', 'benchmark_return_5d_after']

    # Check if any event has identical targets to another (unlikely but possible)
    dup = df.duplicated(subset=target_cols, keep=False).sum()
    return [{
        'target': 'duplicate_check',
        'class': 'DUPLICATE_TARGET_ROWS',
        'split': 'OVERALL',
        'count': int(dup),
        'pct': round(dup / len(df) * 100, 2),
        'check': 'duplicate_targets'
    }]


def check_impossible_values(df, target_cols):
    """Check for impossible target values."""
    results = []

    # Returns should be > -100% (can't lose more than invested)
    for target in target_cols:
        if target.startswith('return_') or target.startswith('abnormal_return_') or target.startswith('benchmark_return_'):
            impossible_low = (df[target] <= -1.0).sum()
            impossible_high = (df[target] > 10.0).sum()  # >1000% in one period is suspicious
            results.append({
                'target': target,
                'class': 'IMPOSSIBLE_LOW',
                'split': 'OVERALL',
                'count': int(impossible_low),
                'pct': round(impossible_low / len(df) * 100, 2),
                'check': 'impossible_values'
            })
            results.append({
                'target': target,
                'class': 'IMPOSSIBLE_HIGH',
                'split': 'OVERALL',
                'count': int(impossible_high),
                'pct': round(impossible_high / len(df) * 100, 2),
                'check': 'impossible_values'
            })

    return results


def main():
    print("Loading data...")
    df, manifest = load_data()

    train_mask, val_mask, test_mask = get_splits(df)
    print(f"Split sizes: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

    target_cols = ['abnormal_return_1d', 'abnormal_return_3d', 'abnormal_return_5d',
                   'return_1d_after', 'return_3d_after', 'return_5d_after',
                   'benchmark_return_1d_after', 'benchmark_return_3d_after', 'benchmark_return_5d_after']

    all_results = []

    # Audit reaction_class
    print("Auditing reaction_class...")
    all_results.extend(audit_reaction_class(df, train_mask, val_mask, test_mask))

    # Audit continuous targets
    print("Auditing continuous targets...")
    all_results.extend(audit_continuous_targets(df, train_mask, val_mask, test_mask, target_cols))

    # Check label consistency
    print("Checking label consistency...")
    all_results.extend(check_label_consistency(df))

    # Check split boundary overlap
    print("Checking split boundary overlap...")
    all_results.extend(check_split_boundary_overlap(df, train_mask, val_mask, test_mask))

    # Check duplicate targets
    print("Checking duplicate targets...")
    all_results.extend(check_duplicate_targets(df))

    # Check impossible values
    print("Checking impossible values...")
    all_results.extend(check_impossible_values(df, target_cols))

    # Save
    results_df = pd.DataFrame(all_results)
    output_path = 'data/processed/target_quality_audit.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Print summary
    print("\n=== TARGET QUALITY SUMMARY ===")
    print(f"Total checks: {len(results_df)}")
    print(f"\nReaction class distribution:")
    rc_dist = df['reaction_class'].value_counts()
    for cls, cnt in rc_dist.items():
        print(f"  {cls}: {cnt} ({cnt/len(df)*100:.1f}%)")

    print(f"\nLabel consistency with ±2% threshold:")
    consistent = results_df[(results_df['target'] == 'reaction_class_consistency') & (results_df['class'] == 'CONSISTENT')]['count'].values
    inconsistent = results_df[(results_df['target'] == 'reaction_class_consistency') & (results_df['class'] == 'INCONSISTENT')]['count'].values
    if len(consistent) > 0:
        print(f"  Consistent: {consistent[0]}")
    if len(inconsistent) > 0:
        print(f"  Inconsistent: {inconsistent[0]}")

    print(f"\nContinuous target stats (overall):")
    for target in target_cols:
        t_data = results_df[(results_df['target'] == target) & (results_df['split'] == 'OVERALL')]
        if len(t_data) > 0:
            r = t_data.iloc[0]
            print(f"  {target}: n={r['count']}, mean={r['mean']:.4f}, std={r['std']:.4f}, "
                  f"min={r['min']:.4f}, max={r['max']:.4f}, missing={r['missing_count']}, inf={r['infinite_count']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())