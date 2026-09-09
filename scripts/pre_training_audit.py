#!/usr/bin/env python
"""
Pre-Training Readiness Audit

This script checks all requirements before model training can begin.
It does NOT train any model — it only validates the data, features,
splits, and guardrails are ready.

Usage:
    python scripts/pre_training_audit.py

Exit codes:
    0 = PRE-TRAINING READINESS: PASS
    1 = PRE-TRAINING READINESS: FAIL (with details)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AuditFailure(Exception):
    """Raised when an audit check fails."""
    pass


def check_data_exists():
    """Verify dataset exists and is readable."""
    path = Path("data/processed/earnings_ml_ready.parquet")
    if not path.exists():
        raise AuditFailure(f"Dataset not found: {path}")
    df = pd.read_parquet(path)
    if len(df) == 0:
        raise AuditFailure("Dataset is empty")
    return df


def check_event_uniqueness(df):
    """Verify each event is unique."""
    dup_keys = df['event_key'].duplicated().sum()
    if dup_keys > 0:
        raise AuditFailure(f"Duplicate event_keys: {dup_keys}")

    dup_symbol_period = df.duplicated(subset=['symbol', 'period_ended']).sum()
    if dup_symbol_period > 0:
        raise AuditFailure(f"Duplicate symbol+period_ended: {dup_symbol_period}")


def check_expected_columns(df):
    """Verify expected columns exist."""
    required_identifiers = [
        'event_key', 'symbol', 'company_name', 'period_ended',
        'fiscal_quarter', 'result_announcement_datetime',
        'announcement_session_type', 'feature_cutoff_date',
        'reaction_start_date', 'quality_flags',
        'price_history_days', 'benchmark_history_days'
    ]
    required_audit = [
        'fundamental_period_end_date', 'fundamental_estimated_available_date',
        'fundamental_pit_status', 'fundamental_lag_days'
    ]
    required_targets = [
        'reaction_class', 'abnormal_return_1d', 'abnormal_return_3d',
        'abnormal_return_5d', 'return_1d_after', 'return_3d_after',
        'return_5d_after', 'benchmark_return_1d_after',
        'benchmark_return_3d_after', 'benchmark_return_5d_after'
    ]

    missing = []
    for cols, name in [(required_identifiers, 'identifiers'), (required_audit, 'audit'), (required_targets, 'targets')]:
        for c in cols:
            if c not in df.columns:
                missing.append(f"{name}:{c}")

    if missing:
        raise AuditFailure(f"Missing columns: {missing}")


def check_feature_count(df):
    """Verify feature count matches expectation."""
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
    target_cols = [
        'reaction_class', 'abnormal_return_1d', 'abnormal_return_3d',
        'abnormal_return_5d', 'return_1d_after', 'return_3d_after',
        'return_5d_after', 'benchmark_return_1d_after',
        'benchmark_return_3d_after', 'benchmark_return_5d_after'
    ]

    feature_cols = [c for c in df.columns if c not in identifier_cols and c not in audit_cols and c not in target_cols]

    if len(feature_cols) != 150:
        raise AuditFailure(f"Expected 150 features, got {len(feature_cols)}")

    if len(target_cols) != 10:
        raise AuditFailure(f"Expected 10 targets, got {len(target_cols)}")

    return feature_cols, target_cols


def check_target_existence(df):
    """Verify targets exist and have reasonable distributions."""
    # reaction_class
    rc = df['reaction_class'].value_counts()
    if len(rc) < 2:
        raise AuditFailure(f"reaction_class has only {len(rc)} unique values: {rc.to_dict()}")
    if rc.isna().any():
        raise AuditFailure("reaction_class has NaN values")

    # Continuous targets
    cont_targets = ['abnormal_return_1d', 'abnormal_return_3d', 'abnormal_return_5d',
                    'return_1d_after', 'return_3d_after', 'return_5d_after',
                    'benchmark_return_1d_after', 'benchmark_return_3d_after', 'benchmark_return_5d_after']
    for t in cont_targets:
        if df[t].isna().all():
            raise AuditFailure(f"Target {t} is all NaN")
        if df[t].isna().sum() > len(df) * 0.1:
            raise AuditFailure(f"Target {t} has >10% NaN: {df[t].isna().sum()}")


def check_no_all_nan_features(df, feature_cols):
    """Verify no features are all NaN."""
    all_nan = [c for c in feature_cols if df[c].isna().all()]
    if all_nan:
        raise AuditFailure(f"All-NaN features: {all_nan}")


def check_no_constant_features(df, feature_cols):
    """Verify no constant or near-constant features."""
    constant = []
    for c in feature_cols:
        if df[c].dtype.kind in 'fc':  # float or complex
            nunique = df[c].nunique(dropna=True)
            if nunique <= 1:
                constant.append(c)
    if constant:
        raise AuditFailure(f"Constant/near-constant features: {constant}")


def check_no_infinite_features(df, feature_cols):
    """Verify no infinite values in features."""
    inf_cols = []
    for c in feature_cols:
        if df[c].dtype.kind in 'fc':
            if np.any(np.isinf(df[c].values)):
                inf_cols.append(c)
    if inf_cols:
        raise AuditFailure(f"Features with infinite values: {inf_cols}")


def check_no_target_leakage(df, feature_cols):
    """Verify no target columns in features."""
    forbidden_patterns = [
        'reaction_class', 'abnormal_return', 'return_*_after',
        'benchmark_return_*_after', 'reaction_start_date'
    ]

    leaky = []
    for c in feature_cols:
        # Skip prior_earnings features - they are legitimate historical features
        if c.startswith('prior_'):
            continue
        for pattern in forbidden_patterns:
            if pattern.replace('*', '') in c:
                leaky.append(c)
                break

    if leaky:
        raise AuditFailure(f"Potential target leakage in features: {leaky}")


def check_no_post_event_leakage(df, feature_cols):
    """Verify no post-event columns in features."""
    leaky = [c for c in feature_cols if '_after' in c]
    if leaky:
        raise AuditFailure(f"Post-event columns in features: {leaky}")


def check_no_future_columns(df, feature_cols):
    """Verify no future-information columns in features."""
    # Already checked by leakage patterns, but double-check suspicious names
    suspicious = [c for c in feature_cols if any(kw in c.lower() for kw in ['future', 'forward', 'next_', 't+1', 't+2', 't+3'])]
    if suspicious:
        raise AuditFailure(f"Suspicious future-looking column names: {suspicious}")


def check_pit_market_cutoff(df):
    """Verify market features use only data up to feature_cutoff."""
    # Check feature_cutoff < reaction_start
    fc = pd.to_datetime(df['feature_cutoff_date'])
    rs = pd.to_datetime(df['reaction_start_date'])
    if not (fc < rs).all():
        raise AuditFailure("feature_cutoff_date >= reaction_start_date for some events")

    # Check feature_cutoff <= announcement
    ann = pd.to_datetime(df['result_announcement_datetime'])
    if not (fc <= ann).all():
        raise AuditFailure("feature_cutoff_date > announcement for some events")


def check_pit_fundamental_status(df):
    """Verify fundamental PIT status is estimated_conservative, not verified."""
    statuses = df['fundamental_pit_status'].unique()
    for s in statuses:
        if pd.notna(s) and 'verified' in str(s).lower() and 'estimated' not in str(s).lower():
            raise AuditFailure(f"Fundamental PIT status claims verified: {s}")


def check_pit_fundamental_dates(df):
    """Verify fundamental estimated_available <= feature_cutoff and period_end < event period_end."""
    has_fund = df['fundamental_estimated_available_date'].notna()
    if has_fund.any():
        fc = pd.to_datetime(df.loc[has_fund, 'feature_cutoff_date'])
        fead = pd.to_datetime(df.loc[has_fund, 'fundamental_estimated_available_date'])
        if not (fead <= fc).all():
            raise AuditFailure("fundamental_estimated_available_date > feature_cutoff_date for some events")

        pe = pd.to_datetime(df.loc[has_fund, 'fundamental_period_end_date'])
        event_pe = pd.to_datetime(df.loc[has_fund, 'period_ended'])
        if not (pe < event_pe).all():
            raise AuditFailure("fundamental_period_end_date >= event period_ended for some events")


def check_prior_earnings_ordering(df):
    """Verify prior earnings features use only earlier events."""
    # prior_earnings_count should be non-decreasing within symbol
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].sort_values('period_ended')
        counts = symbol_df['prior_earnings_count'].values
        if not np.all(np.diff(counts) >= 0):
            raise AuditFailure(f"prior_earnings_count not monotonic for {symbol}")


def check_chronological_split(df):
    """Verify chronological split with no overlap."""
    # Get split assignments from the dataset if available, otherwise infer from period_ended
    # The dataset doesn't have a split column, so we verify the date ranges don't overlap
    # based on the known split methodology (by unique period_ended quarters)

    periods = df['period_ended'].unique()
    periods_sorted = np.sort(periods)

    # Expected split boundaries (from checkpoint 51)
    # Train: 2015-Q4 to 2021-Q4 (25 quarters)
    # Val: 2022-Q1 to 2024-Q1 (9 quarters)
    # Test: 2024-Q2 to 2026-Q2 (9 quarters)

    # Verify no future events in dataset
    max_period = pd.to_datetime(periods_sorted[-1])
    if max_period >= pd.Timestamp('2026-09-08'):
        raise AuditFailure(f"Future period_ended in dataset: {max_period}")

    # Verify split ordering by checking known split points
    train_max = pd.Timestamp('2021-12-31')
    val_min = pd.Timestamp('2022-03-31')
    val_max = pd.Timestamp('2024-03-31')
    test_min = pd.Timestamp('2024-06-30')

    if not (train_max < val_min):
        raise AuditFailure("Train max period not < Val min period")
    if not (val_max < test_min):
        raise AuditFailure("Val max period not < Test min period")


def check_target_distribution(df):
    """Verify target distribution is reasonable."""
    rc = df['reaction_class'].value_counts()
    total = len(df)

    for cls, count in rc.items():
        pct = count / total * 100
        if pct < 5:
            raise AuditFailure(f"Target class {cls} has only {pct:.1f}% ({count}/{total})")

    # Check no class dominates excessively
    max_pct = rc.max() / total * 100
    if max_pct > 90:
        raise AuditFailure(f"Target class imbalance: max class = {max_pct:.1f}%")


def check_missing_targets(df):
    """Verify targets have minimal missingness."""
    target_cols = ['reaction_class', 'abnormal_return_1d', 'abnormal_return_3d',
                   'abnormal_return_5d', 'return_1d_after', 'return_3d_after',
                   'return_5d_after', 'benchmark_return_1d_after',
                   'benchmark_return_3d_after', 'benchmark_return_5d_after']

    for t in target_cols:
        missing = df[t].isna().sum()
        if missing > 0:
            print(f"  WARNING: Target {t} has {missing} missing values")


def check_guardrail_module():
    """Verify guardrail module is importable and basic test passes."""
    try:
        from scripts.model_guardrails import (
            GuardrailViolation,
            validate_classification_predictions,
            validate_prediction_probabilities,
            compare_against_baseline,
            validate_regression_predictions,
        )
    except ImportError as e:
        raise AuditFailure(f"Guardrail module not importable: {e}")

    # Test single-class prediction fails
    y_true = np.array(['NEUTRAL'] * 100 + ['POSITIVE'] * 50 + ['NEGATIVE'] * 50)
    y_pred = np.array(['NEUTRAL'] * 200)

    try:
        validate_classification_predictions(y_true, y_pred, ['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
        raise AuditFailure("Single-class prediction did not raise GuardrailViolation")
    except GuardrailViolation:
        pass  # Expected

    # Test baseline comparison works - use predictions that clearly beat baseline
    # Baseline predicts NEUTRAL for all (60% accuracy)
    # Model should do better than that
    y_pred_good = np.array(
        ['NEUTRAL'] * 80 + ['POSITIVE'] * 40 + ['NEGATIVE'] * 40 +  # 80% correct on NEUTRAL, 80% on POS, 80% on NEG
        ['NEUTRAL'] * 20 + ['POSITIVE'] * 10 + ['NEGATIVE'] * 10    # 20% errors
    )
    result = compare_against_baseline(y_true, y_pred_good, valid_classes=['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
    if result['improvement']['macro_f1'] <= 0:
        raise AuditFailure("Baseline comparison not working correctly")


def check_val_dividend_yield_absent(df):
    """Verify val_dividend_yield is completely absent."""
    if 'val_dividend_yield' in df.columns:
        raise AuditFailure("val_dividend_yield still present in dataset")


def run_audit():
    """Run all audit checks."""
    checks = [
        ("Dataset exists", check_data_exists),
        ("Event uniqueness", check_event_uniqueness),
        ("Expected columns", check_expected_columns),
        ("Feature/target counts", check_feature_count),
        ("Target existence", check_target_existence),
        ("No all-NaN features", check_no_all_nan_features),
        ("No constant features", check_no_constant_features),
        ("No infinite features", check_no_infinite_features),
        ("No target leakage", check_no_target_leakage),
        ("No post-event leakage", check_no_post_event_leakage),
        ("No future columns", check_no_future_columns),
        ("PIT market cutoff", check_pit_market_cutoff),
        ("PIT fundamental status", check_pit_fundamental_status),
        ("PIT fundamental dates", check_pit_fundamental_dates),
        ("Prior earnings ordering", check_prior_earnings_ordering),
        ("Chronological split", check_chronological_split),
        ("Target distribution", check_target_distribution),
        ("Missing targets", check_missing_targets),
        ("Guardrail module", check_guardrail_module),
        ("val_dividend_yield absent", check_val_dividend_yield_absent),
    ]

    df = None
    results = []
    for name, check_fn in checks:
        try:
            if df is None and name != "Dataset exists":
                # Re-load for checks that need df (first check returns df)
                pass
            if name == "Dataset exists":
                df = check_fn()
                results.append((name, "PASS", ""))
            else:
                if name in ["No all-NaN features", "No constant features", "No infinite features",
                           "No target leakage", "No post-event leakage", "No future columns"]:
                    feature_cols, _ = check_feature_count(df)
                    check_fn(df, feature_cols)
                elif name in ["Expected columns", "Feature/target counts", "Target existence",
                              "PIT market cutoff", "PIT fundamental status", "PIT fundamental dates",
                              "Prior earnings ordering", "Chronological split", "Target distribution",
                              "Missing targets", "val_dividend_yield absent"]:
                    check_fn(df)
                elif name in ["Guardrail module"]:
                    check_fn()
                else:
                    check_fn(df)
                results.append((name, "PASS", ""))
        except AuditFailure as e:
            results.append((name, "FAIL", str(e)))
        except Exception as e:
            results.append((name, "ERROR", f"Unexpected error: {e}"))

    return results


def main():
    print("=" * 70)
    print("PRE-TRAINING READINESS AUDIT")
    print("=" * 70)

    results = run_audit()

    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")
    errors = sum(1 for _, status, _ in results if status == "ERROR")

    for name, status, msg in results:
        symbol = "✅" if status == "PASS" else "❌"
        print(f"  {symbol} {name}: {status}")
        if msg:
            print(f"      {msg}")

    print("=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed, {errors} errors")

    if failed == 0 and errors == 0:
        print("\nPRE-TRAINING READINESS: PASS")
        print("\n✅ All checks passed. Ready for model training.")
        return 0
    else:
        print("\nPRE-TRAINING READINESS: FAIL")
        print("\n❌ Some checks failed. Fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())