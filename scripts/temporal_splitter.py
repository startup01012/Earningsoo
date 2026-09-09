#!/usr/bin/env python
"""
Temporal Splitter with 5-Trading-Day Embargo Enforcement

This module provides reusable temporal splitting utilities that enforce
a 5-trading-day embargo between train/validation and validation/test splits
to prevent label-window contamination.

The maximum target horizon is 5 trading days, so we must ensure that:
- The last training label window does not overlap the validation information period
- The last validation label window does not overlap the test information period

Uses a proper trading day calendar (weekdays excluding major holidays) to
calculate embargo periods, not just dates present in the dataset.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SplitResult:
    """Result of a temporal split."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    embargo_days: int
    metadata: Dict[str, Any]


@dataclass 
class WalkForwardFold:
    """Single fold in walk-forward validation."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    embargo_start: pd.Timestamp
    embargo_end: pd.Timestamp
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_rows: int
    val_rows: int


def generate_trading_day_calendar(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    holidays: Optional[List[pd.Timestamp]] = None
) -> pd.DatetimeIndex:
    """
    Generate a trading day calendar (weekdays excluding holidays).
    
    Args:
        start_date: Start of calendar
        end_date: End of calendar
        holidays: Optional list of holiday dates to exclude
        
    Returns:
        DatetimeIndex of trading days
    """
    # Generate all weekdays (Monday=0, Friday=4)
    all_days = pd.date_range(start=start_date, end=end_date, freq='D')
    weekdays = all_days[all_days.weekday < 5]  # Mon-Fri
    
    if holidays:
        holidays = pd.DatetimeIndex(holidays)
        weekdays = weekdays.difference(holidays)
    
    return weekdays


def get_trading_day_calendar_from_data(
    df: pd.DataFrame,
    date_column: str = 'feature_cutoff_date',
    buffer_days: int = 30
) -> pd.DatetimeIndex:
    """
    Generate a trading day calendar spanning the dataset date range.
    
    Args:
        df: Dataset with date column
        date_column: Column containing dates
        buffer_days: Extra days to add before/after data range
        
    Returns:
        DatetimeIndex of trading days covering the data range
    """
    dates = pd.to_datetime(df[date_column]).dropna()
    start = dates.min() - pd.Timedelta(days=buffer_days)
    end = dates.max() + pd.Timedelta(days=buffer_days)
    
    return generate_trading_day_calendar(start, end)


def add_embargo_trading_days(
    split_date: pd.Timestamp,
    trading_days: pd.DatetimeIndex,
    embargo_days: int = 5
) -> pd.Timestamp:
    """
    Add embargo trading days to a split point.
    
    Args:
        split_date: The original split date
        trading_days: Trading day calendar
        embargo_days: Number of trading days to add as embargo
        
    Returns:
        The embargo-adjusted date (first trading day after embargo)
    """
    # Find the first trading day strictly after split_date
    after_split = trading_days[trading_days > split_date]
    if len(after_split) == 0:
        return trading_days[-1]
    
    # Start from the first trading day after split
    start_pos = trading_days.get_loc(after_split[0])
    
    # Move forward by embargo_days - 1 (since we start at first day after split)
    embargo_pos = start_pos + embargo_days - 1
    
    if embargo_pos >= len(trading_days):
        return trading_days[-1]
    
    return trading_days[embargo_pos]


def count_trading_days_between(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    trading_days: pd.DatetimeIndex
) -> int:
    """
    Count trading days strictly between two dates.
    
    Args:
        start_date: Start date (exclusive)
        end_date: End date (exclusive)
        trading_days: Trading day calendar
        
    Returns:
        Number of trading days strictly between start and end
    """
    between = trading_days[(trading_days > start_date) & (trading_days < end_date)]
    return len(between)


def create_chronological_split_with_embargo(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    embargo_days: int = 5,
    date_column: str = 'period_ended',
    trading_day_column: str = 'feature_cutoff_date'
) -> SplitResult:
    """
    Create chronological train/val/test splits with trading-day embargo.
    
    The embargo ensures that validation feature_cutoff dates are at least
    `embargo_days` trading days after the last training feature_cutoff date.
    
    Args:
        df: Full dataset
        train_end: Last period_ended for training
        val_end: Last period_ended for validation
        embargo_days: Number of trading days for embargo
        date_column: Column to split on (typically period_ended)
        trading_day_column: Column for feature cutoff dates
        
    Returns:
        SplitResult with train/val/test dataframes and metadata
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df[trading_day_column] = pd.to_datetime(df[trading_day_column])
    
    # Generate trading day calendar from data
    trading_days = get_trading_day_calendar_from_data(df, trading_day_column)
    
    # Initial split based on period_ended
    train_mask = df[date_column] <= train_end
    val_mask = (df[date_column] > train_end) & (df[date_column] <= val_end)
    test_mask = df[date_column] > val_end
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    # Get feature cutoff ranges for each split
    train_max_fc = pd.to_datetime(train_df[trading_day_column]).max()
    val_min_fc = pd.to_datetime(val_df[trading_day_column]).min()
    val_max_fc = pd.to_datetime(val_df[trading_day_column]).max()
    test_min_fc = pd.to_datetime(test_df[trading_day_column]).min()
    
    # Apply embargo: ensure val_min_fc is at least embargo_days after train_max_fc
    required_val_start = add_embargo_trading_days(train_max_fc, trading_days, embargo_days)
    
    if val_min_fc < required_val_start:
        # Need to move some events from val to test to satisfy embargo
        val_mask = val_mask & (df[trading_day_column] >= required_val_start)
        val_df = df[val_mask].copy()
        val_min_fc = pd.to_datetime(val_df[trading_day_column]).min()
        val_max_fc = pd.to_datetime(val_df[trading_day_column]).max()
    
    # Apply embargo between val and test
    required_test_start = add_embargo_trading_days(val_max_fc, trading_days, embargo_days)
    
    if test_min_fc < required_test_start:
        test_mask = test_mask & (df[trading_day_column] >= required_test_start)
        test_df = df[test_mask].copy()
        test_min_fc = pd.to_datetime(test_df[trading_day_column]).min()
    
    # Recompute masks after embargo adjustment
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    # Verify no overlap
    train_periods = set(train_df[date_column].unique())
    val_periods = set(val_df[date_column].unique())
    test_periods = set(test_df[date_column].unique())
    
    # Verify chronological order (by period_ended)
    assert train_df[date_column].max() < val_df[date_column].min(), "Train not before Val"
    assert val_df[date_column].max() < test_df[date_column].min(), "Val not before Test"
    
    # Verify embargo is respected (by feature_cutoff dates)
    train_max_fc_final = pd.to_datetime(train_df[trading_day_column]).max()
    val_min_fc_final = pd.to_datetime(val_df[trading_day_column]).min()
    val_max_fc_final = pd.to_datetime(val_df[trading_day_column]).max()
    test_min_fc_final = pd.to_datetime(test_df[trading_day_column]).min()
    
    train_val_gap = count_trading_days_between(train_max_fc_final, val_min_fc_final, trading_days)
    val_test_gap = count_trading_days_between(val_max_fc_final, test_min_fc_final, trading_days)
    
    metadata = {
        'split_type': 'chronological_with_embargo',
        'train_end': train_end,
        'val_end': val_end,
        'embargo_days': embargo_days,
        'train_max_feature_cutoff': train_max_fc_final,
        'val_min_feature_cutoff': val_min_fc_final,
        'val_max_feature_cutoff': val_max_fc_final,
        'test_min_feature_cutoff': test_min_fc_final,
        'train_val_trading_day_gap': train_val_gap,
        'val_test_trading_day_gap': val_test_gap,
        'embargo_satisfied_train_val': train_val_gap >= embargo_days,
        'embargo_satisfied_val_test': val_test_gap >= embargo_days,
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
    }
    
    return SplitResult(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        embargo_days=embargo_days,
        metadata=metadata
    )


def create_walk_forward_splits(
    df: pd.DataFrame,
    initial_train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    step_months: int = 3,
    embargo_days: int = 5,
    date_column: str = 'period_ended',
    trading_day_column: str = 'feature_cutoff_date',
    min_train_periods: int = 4
) -> List[WalkForwardFold]:
    """
    Create expanding window walk-forward splits with embargo.
    
    Each fold expands the training window by step_months and validates
    on the next period, with embargo between train and validation.
    
    Args:
        df: Full dataset
        initial_train_end: Initial training end date
        val_end: Final validation end date
        step_months: Months to expand training window each fold
        embargo_days: Trading days embargo
        date_column: Column to split on
        trading_day_column: Column for trading day calendar
        min_train_periods: Minimum unique periods in training
        
    Returns:
        List of WalkForwardFold objects
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df[trading_day_column] = pd.to_datetime(df[trading_day_column])
    
    trading_days = get_trading_day_calendar_from_data(df, trading_day_column)
    
    folds = []
    fold_id = 0
    current_train_end = initial_train_end
    
    while True:
        # Validation period: next step_months after current_train_end
        val_start_candidate = current_train_end + pd.DateOffset(months=1)
        val_end_candidate = min(
            val_start_candidate + pd.DateOffset(months=step_months) - pd.DateOffset(days=1), 
            val_end
        )
        
        if val_start_candidate > val_end:
            break
        
        # Apply embargo to determine actual validation start
        train_mask = df[date_column] <= current_train_end
        train_max_fc = pd.to_datetime(df.loc[train_mask, trading_day_column]).max()
        required_val_start = add_embargo_trading_days(train_max_fc, trading_days, embargo_days)
        
        val_mask = (df[date_column] > required_val_start) & (df[date_column] <= val_end_candidate)
        val_indices = np.where(val_mask)[0]
        train_indices = np.where(train_mask)[0]
        
        if len(val_indices) == 0 or len(train_indices) == 0:
            current_train_end += pd.DateOffset(months=step_months)
            continue
            
        # Check minimum training periods
        train_periods = df.loc[train_mask, date_column].nunique()
        if train_periods < min_train_periods:
            current_train_end += pd.DateOffset(months=step_months)
            continue
        
        # Verify embargo
        train_max_fc_val = pd.to_datetime(df.loc[train_mask, trading_day_column]).max()
        val_min_fc_val = pd.to_datetime(df.loc[val_mask, trading_day_column]).min()
        gap = count_trading_days_between(train_max_fc_val, val_min_fc_val, trading_days)
        
        if gap < embargo_days:
            current_train_end += pd.DateOffset(months=step_months)
            continue
        
        # Verify no overlap
        assert df.loc[train_mask, date_column].max() < df.loc[val_mask, date_column].min()
        
        fold = WalkForwardFold(
            fold_id=fold_id,
            train_start=pd.to_datetime(df.loc[train_mask, date_column]).min(),
            train_end=pd.to_datetime(df.loc[train_mask, date_column]).max(),
            val_start=pd.to_datetime(df.loc[val_mask, date_column]).min(),
            val_end=pd.to_datetime(df.loc[val_mask, date_column]).max(),
            embargo_start=train_max_fc_val,
            embargo_end=val_min_fc_val,
            train_indices=train_indices,
            val_indices=val_indices,
            train_rows=len(train_indices),
            val_rows=len(val_indices)
        )
        folds.append(fold)
        fold_id += 1
        
        # Expand training window
        current_train_end += pd.DateOffset(months=step_months)
        
        if current_train_end >= val_end:
            break
    
    return folds


def verify_split_embargo(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trading_day_column: str = 'feature_cutoff_date',
    embargo_days: int = 5
) -> Dict[str, Any]:
    """
    Verify that a split respects the embargo.
    
    Returns:
        Dictionary with verification results
    """
    trading_days = get_trading_day_calendar_from_data(
        pd.concat([train_df, val_df, test_df]), trading_day_column
    )
    
    train_max = pd.to_datetime(train_df[trading_day_column]).max()
    val_min = pd.to_datetime(val_df[trading_day_column]).min()
    val_max = pd.to_datetime(val_df[trading_day_column]).max()
    test_min = pd.to_datetime(test_df[trading_day_column]).min()
    
    results = {
        'train_max_fc': train_max,
        'val_min_fc': val_min,
        'val_max_fc': val_max,
        'test_min_fc': test_min,
        'embargo_days_required': embargo_days
    }
    
    if pd.notna(train_max) and pd.notna(val_min):
        gap = count_trading_days_between(train_max, val_min, trading_days)
        results['train_val_trading_days_between'] = gap
        results['train_val_embargo_ok'] = gap >= embargo_days
    else:
        results['train_val_trading_days_between'] = None
        results['train_val_embargo_ok'] = False
        
    if pd.notna(val_max) and pd.notna(test_min):
        gap = count_trading_days_between(val_max, test_min, trading_days)
        results['val_test_trading_days_between'] = gap
        results['val_test_embargo_ok'] = gap >= embargo_days
    else:
        results['val_test_trading_days_between'] = None
        results['val_test_embargo_ok'] = False
        
    return results


if __name__ == "__main__":
    # Test with actual dataset
    import pandas as pd
    
    print("Testing temporal splitter with embargo...")
    df = pd.read_parquet("data/processed/earnings_ml_ready.parquet")
    
    # Test chronological split with embargo
    result = create_chronological_split_with_embargo(
        df,
        train_end=pd.Timestamp('2021-12-31'),
        val_end=pd.Timestamp('2024-03-31'),
        embargo_days=5
    )
    
    print(f"Train rows: {result.metadata['train_rows']}")
    print(f"Val rows: {result.metadata['val_rows']}")
    print(f"Test rows: {result.metadata['test_rows']}")
    print(f"Train max FC: {result.metadata['train_max_feature_cutoff']}")
    print(f"Val min FC: {result.metadata['val_min_feature_cutoff']}")
    print(f"Val max FC: {result.metadata['val_max_feature_cutoff']}")
    print(f"Test min FC: {result.metadata['test_min_feature_cutoff']}")
    print(f"Train/Val trading day gap: {result.metadata['train_val_trading_day_gap']}")
    print(f"Val/Test trading day gap: {result.metadata['val_test_trading_day_gap']}")
    print(f"Embargo satisfied (train/val): {result.metadata['embargo_satisfied_train_val']}")
    print(f"Embargo satisfied (val/test): {result.metadata['embargo_satisfied_val_test']}")
    
    # Verify
    verification = verify_split_embargo(
        result.train_df, result.val_df, result.test_df,
        embargo_days=5
    )
    print(f"Verification: {verification}")
    
    # Test walk-forward
    print("\nTesting walk-forward splits...")
    folds = create_walk_forward_splits(
        df,
        initial_train_end=pd.Timestamp('2021-12-31'),
        val_end=pd.Timestamp('2024-03-31'),
        step_months=3,
        embargo_days=5
    )
    
    for fold in folds[:5]:
        print(f"Fold {fold.fold_id}: train={fold.train_rows}, val={fold.val_rows}, "
              f"train [{fold.train_start.date()} - {fold.train_end.date()}], "
              f"val [{fold.val_start.date()} - {fold.val_end.date()}]")
    
    print("✅ Temporal splitter tests passed!")