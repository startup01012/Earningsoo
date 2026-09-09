#!/usr/bin/env python
"""
Missingness Strategy Audit

Analyzes missingness patterns across TRAIN/VALIDATION/TEST splits for every feature family.
Determines:
- Whether missing values are expected
- Whether missingness itself could contain information
- Whether missingness changes substantially over time
- Whether missingness differs between train/validation/test

Designs preprocessing contract for future training phase.

Outputs:
- data/processed/missingness_audit.csv
- data/processed/missingness_strategy.md
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
    train_max = pd.Timestamp('2021-12-31')
    val_max = pd.Timestamp('2024-03-31')

    df['period_ended_dt'] = pd.to_datetime(df['period_ended'])

    train_mask = df['period_ended_dt'] <= train_max
    val_mask = (df['period_ended_dt'] > train_max) & (df['period_ended_dt'] <= val_max)
    test_mask = df['period_ended_dt'] > val_max

    return train_mask, val_mask, test_mask


def analyze_missingness(df, feature_cols, manifest, train_mask, val_mask, test_mask):
    """Analyze missingness for each feature across splits."""

    # Merge with manifest for feature group info
    manifest_features = manifest[manifest['is_target'] == False][['feature_name', 'feature_group']].copy()

    results = []
    for col in feature_cols:
        series = df[col]
        train_series = series[train_mask]
        val_series = series[val_mask]
        test_series = series[test_mask]

        overall_missing = series.isna().mean()
        train_missing = train_series.isna().mean()
        val_missing = val_series.isna().mean()
        test_missing = test_series.isna().mean()

        # Missingness difference between splits
        max_miss_diff = max(train_missing, val_missing, test_missing) - min(train_missing, val_missing, test_missing)

        # Missingness trend over time (correlation with period_ended)
        period_ended = pd.to_datetime(df['period_ended'])
        missing_indicator = series.isna().astype(int)
        # Only compute if there's variation
        if missing_indicator.nunique() > 1:
            try:
                miss_corr = missing_indicator.corr(period_ended.astype('int64'))
            except Exception:
                miss_corr = np.nan
        else:
            miss_corr = np.nan

        # Feature group
        group_row = manifest_features[manifest_features['feature_name'] == col]
        feature_group = group_row['feature_group'].values[0] if len(group_row) > 0 else 'UNKNOWN'

        # Determine missingness type
        if feature_group in ['fundamental_balance', 'fundamental_income', 'fundamental_derived', 'fundamental_cashflow']:
            missingness_type = 'EXPECTED_UNAVAILABLE'
            missingness_reason = 'Fundamental data only available ~2022+ from yfinance; pre-2022 events have no fundamentals'
        elif feature_group == 'valuation':
            missingness_type = 'EXPECTED_UNAVAILABLE'
            missingness_reason = 'Valuation features require both price and fundamentals; missing when fundamentals unavailable'
        elif feature_group == 'prior_earnings':
            missingness_type = 'EXPECTED_STRUCTURAL'
            missingness_reason = 'Prior earnings features require historical events; early events have fewer priors'
        else:
            missingness_type = 'UNEXPECTED'
            missingness_reason = 'Market features should have near-complete coverage'

        # Could missingness be informative?
        informative_missing = False
        if feature_group == 'prior_earnings':
            informative_missing = True  # Missing = no prior history = early in symbol's life
        elif feature_group in ['fundamental_balance', 'fundamental_income', 'fundamental_derived', 'fundamental_cashflow', 'valuation']:
            informative_missing = True  # Missing = pre-2022 or no yfinance coverage

        results.append({
            'feature_name': col,
            'feature_group': feature_group,
            'overall_missing_pct': round(overall_missing * 100, 2),
            'train_missing_pct': round(train_missing * 100, 2),
            'validation_missing_pct': round(val_missing * 100, 2),
            'test_missing_pct': round(test_missing * 100, 2),
            'max_missingness_diff_pct': round(max_miss_diff * 100, 2),
            'missingness_time_correlation': round(miss_corr, 4) if not np.isnan(miss_corr) else np.nan,
            'missingness_type': missingness_type,
            'missingness_reason': missingness_reason,
            'informative_missing': informative_missing,
            'missing_indicates_zero': False,  # Financial features - missing != zero
        })

    return pd.DataFrame(results)


def generate_strategy_doc(missingness_df):
    """Generate the missingness strategy markdown document."""

    md = """# Missingness Strategy for EarningsOS ML Pipeline

## Overview

This document defines the preprocessing contract for handling missing values in the EarningsOS ML dataset.
The dataset has **150 features** across 4 families with varying missingness patterns.

## Missingness by Feature Family

"""

    # Summary by group
    group_summary = missingness_df.groupby('feature_group').agg(
        n_features=('feature_name', 'count'),
        avg_missing=('overall_missing_pct', 'mean'),
        max_missing=('overall_missing_pct', 'max'),
        min_missing=('overall_missing_pct', 'min'),
        informative_missing=('informative_missing', 'any')
    ).reset_index()

    md += "| Feature Group | N Features | Avg Missing % | Max Missing % | Min Missing % | Informative Missing |\n"
    md += "|---------------|------------|---------------|---------------|---------------|---------------------|\n"
    for _, row in group_summary.iterrows():
        md += f"| {row['feature_group']} | {row['n_features']} | {row['avg_missing']:.1f} | {row['max_missing']:.1f} | {row['min_missing']:.1f} | {row['informative_missing']} |\n"

    md += """

## Missingness Classification

| Type | Description | Action |
|------|-------------|--------|
| `EXPECTED_UNAVAILABLE` | Data source doesn't exist for period (e.g., yfinance fundamentals pre-2022) | Missingness is informative; add missing indicator |
| `EXPECTED_STRUCTURAL` | Structural by design (e.g., no prior earnings for early events) | Missingness is informative; add missing indicator |
| `UNEXPECTED` | Should be available but isn't (e.g., market data gaps) | Investigate; may indicate data quality issue |

## Preprocessing Contract

### For Tree-Based Models (LightGBM, XGBoost, CatBoost, RandomForest)

1. **Preserve NaN natively** - Tree models handle NaN natively via optimal split direction
2. **Add missing indicators** for `EXPECTED_UNAVAILABLE` and `EXPECTED_STRUCTURAL` features:
   - Create binary column `{feature_name}_was_missing` = 1 if original was NaN
   - This allows model to learn "missing = informative" patterns
3. **Never impute with zero** for financial features - zero has economic meaning (e.g., zero revenue, zero debt)
4. **Do NOT use global imputation** (mean/median) - leaks information across time

### For Linear Models (LogisticRegression, Ridge, LinearRegression)

1. **Train-only imputation** - Fit imputer on training data only
2. **Strategy by missingness type**:
   - `EXPECTED_UNAVAILABLE` / `EXPECTED_STRUCTURAL`: Impute with **median of non-missing in train** + add missing indicator
   - `UNEXPECTED`: Impute with **median of non-missing in train** (no indicator needed if rare)
3. **Missing indicators** - Always add binary indicator for features with >5% missing in train
4. **Standardize after imputation** - Fit scaler on imputed training data only

### For All Models

1. **No data leakage** - All preprocessing fitted on TRAIN only, applied to VAL/TEST
2. **Missing ≠ Zero** - Never treat missing financial values as zero
   - Missing revenue ≠ zero revenue
   - Missing debt ≠ zero debt
   - Missing prior earnings ≠ no prior earnings (could be early in history)
3. **Time-aware validation** - When evaluating imputation strategies, use walk-forward validation

## Feature-Specific Recommendations

"""

    # Add feature-specific recommendations
    for group in missingness_df['feature_group'].unique():
        group_df = missingness_df[missingness_df['feature_group'] == group]
        md += f"\n### {group}\n\n"
        md += "| Feature | Missing % | Type | Recommendation |\n"
        md += "|---------|-----------|------|----------------|\n"
        for _, row in group_df.iterrows():
            if row['overall_missing_pct'] > 5:
                rec = "Add missing indicator + train-median imputation (linear); native NaN (tree)"
                if row['informative_missing']:
                    rec = "Add missing indicator (highly informative) + train-median imputation (linear); native NaN (tree)"
                md += f"| {row['feature_name']} | {row['overall_missing_pct']:.1f}% | {row['missingness_type']} | {rec} |\n"

    md += """

## Train/Validation/Test Missingness Consistency

The following features show substantial missingness differences between splits (>10% absolute difference):

"""

    shift_features = missingness_df[missingness_df['max_missingness_diff_pct'] > 10]
    if len(shift_features) > 0:
        md += "| Feature | Train % | Val % | Test % | Max Diff % |\n"
        md += "|---------|---------|-------|--------|------------|\n"
        for _, row in shift_features.iterrows():
            md += f"| {row['feature_name']} | {row['train_missing_pct']:.1f} | {row['validation_missing_pct']:.1f} | {row['test_missing_pct']:.1f} | {row['max_missingness_diff_pct']:.1f} |\n"
    else:
        md += "None (all features have <10% missingness difference between splits)\n"

    md += """

## Temporal Missingness Trends

Features with significant correlation between missingness and time (|corr| > 0.3):

"""

    trend_features = missingness_df[missingness_df['missingness_time_correlation'].abs() > 0.3]
    if len(trend_features) > 0:
        md += "| Feature | Correlation with Time |\n"
        md += "|---------|----------------------|\n"
        for _, row in trend_features.iterrows():
            md += f"| {row['feature_name']} | {row['missingness_time_correlation']:.3f} |\n"
    else:
        md += "None\n"

    md += """

## Key Principles

1. **Missing financial data ≠ Zero** - This is the most critical rule. Zero revenue, zero debt, zero cash flow have specific economic meanings. Missing means "we don't know" or "not reported yet".

2. **Missingness is often informative** - For fundamentals, missing = pre-2022 (no yfinance data). For prior earnings, missing = early in symbol's public history. These are predictive signals.

3. **Never impute across time** - Using future data to impute past missing values leaks information. Always fit imputers on training data only.

4. **Tree models prefer native NaN** - LightGBM/XGBoost/CatBoost handle NaN optimally. Adding indicators helps but isn't strictly required.

5. **Linear models need explicit handling** - Must impute + add indicators. Use train-only statistics.

## Implementation Checklist for Training Pipeline

- [ ] Missing indicator columns added for all features with >5% missing in train
- [ ] Imputer fitted on train only (median for continuous, mode for categorical)
- [ ] Scaler fitted on imputed train only
- [ ] Same transformations applied to validation and test
- [ ] No feature uses test/validation statistics for imputation
- [ ] Missing indicators preserved through pipeline
- [ ] Documentation of which features got indicators and why

"""

    return md


def main():
    print("Loading data...")
    df, manifest = load_data()

    feature_cols, target_cols, identifier_cols, audit_cols = get_feature_columns(df, manifest)
    print(f"Found {len(feature_cols)} feature columns")

    train_mask, val_mask, test_mask = get_splits(df)
    print(f"Split sizes: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

    # Analyze missingness
    print("Analyzing missingness...")
    missingness_df = analyze_missingness(df, feature_cols, manifest, train_mask, val_mask, test_mask)

    # Save CSV
    csv_path = 'data/processed/missingness_audit.csv'
    missingness_df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Generate strategy document
    print("Generating strategy document...")
    md = generate_strategy_doc(missingness_df)
    md_path = 'data/processed/missingness_strategy.md'
    with open(md_path, 'w') as f:
        f.write(md)
    print(f"Saved strategy to {md_path}")

    # Print summary
    print("\n=== MISSINGNESS SUMMARY ===")
    print(missingness_df.groupby('feature_group').agg(
        n=('feature_name', 'count'),
        avg_miss=('overall_missing_pct', 'mean'),
        max_miss=('overall_missing_pct', 'max'),
        informative=('informative_missing', 'any')
    ).to_string())

    # Features with >10% split difference
    shift = missingness_df[missingness_df['max_missingness_diff_pct'] > 10]
    if len(shift) > 0:
        print(f"\n=== FEATURES WITH >10% SPLIT MISSINGNESS DIFFERENCE ({len(shift)}) ===")
        for _, r in shift.iterrows():
            print(f"  {r['feature_name']}: train={r['train_missing_pct']:.1f}%, val={r['validation_missing_pct']:.1f}%, test={r['test_missing_pct']:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())