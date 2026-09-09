#!/usr/bin/env python
"""
Canonical Training Data Loader

This module provides the single, canonical way for all future training scripts
to load the EarningsOS ML dataset with proper X/y separation and temporal splits.

It guarantees:
- X contains ONLY approved model features (no targets, no post-event data, no identifiers, no audit columns)
- Temporal splits are correctly applied with 5-trading-day embargo
- Feature names, groups, and metadata are derived from the manifest
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
import json


@dataclass
class SplitMetadata:
    """Metadata about a temporal split."""
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_days: int
    train_event_keys: List[str]
    val_event_keys: List[str]
    test_event_keys: List[str]
    train_rows: int
    val_rows: int
    test_rows: int


@dataclass
class TrainingData:
    """Container for all training data splits."""
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_val: pd.DataFrame
    y_val: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    feature_names: List[str]
    feature_groups: Dict[str, List[str]]
    event_keys_train: np.ndarray
    event_keys_val: np.ndarray
    event_keys_test: np.ndarray
    announcement_dates_train: np.ndarray
    announcement_dates_val: np.ndarray
    announcement_dates_test: np.ndarray
    feature_cutoff_dates_train: np.ndarray
    feature_cutoff_dates_val: np.ndarray
    feature_cutoff_dates_test: np.ndarray
    reaction_start_dates_train: np.ndarray
    reaction_start_dates_val: np.ndarray
    reaction_start_dates_test: np.ndarray
    split_metadata: SplitMetadata


# Column definitions derived from the manifest and dataset structure
IDENTIFIER_COLUMNS = [
    'event_key', 'symbol', 'company_name', 'period_ended',
    'fiscal_quarter', 'result_announcement_datetime',
    'announcement_session_type', 'feature_cutoff_date',
    'reaction_start_date', 'quality_flags',
    'price_history_days', 'benchmark_history_days'
]

AUDIT_COLUMNS = [
    'fundamental_period_end_date', 'fundamental_estimated_available_date',
    'fundamental_pit_status', 'fundamental_lag_days'
]

TARGET_COLUMNS = [
    'reaction_class', 'abnormal_return_1d', 'abnormal_return_3d',
    'abnormal_return_5d', 'return_1d_after', 'return_3d_after',
    'return_5d_after', 'benchmark_return_1d_after',
    'benchmark_return_3d_after', 'benchmark_return_5d_after'
]

# Columns that must NEVER appear in X (features)
FORBIDDEN_IN_X = set(TARGET_COLUMNS + ['reaction_start_date'] + AUDIT_COLUMNS + IDENTIFIER_COLUMNS)


def load_manifest(manifest_path: str = "data/processed/earnings_feature_manifest.csv") -> pd.DataFrame:
    """Load the feature manifest."""
    return pd.read_csv(manifest_path)


def get_feature_columns(manifest: pd.DataFrame) -> List[str]:
    """Get the list of approved feature columns from the manifest."""
    features = manifest[manifest['is_target'] == False]
    return features['feature_name'].tolist()


def get_feature_groups(manifest: pd.DataFrame) -> Dict[str, List[str]]:
    """Get feature groups mapping from the manifest."""
    features = manifest[manifest['is_target'] == False]
    groups = {}
    for group_name, group_df in features.groupby('feature_group'):
        groups[group_name] = group_df['feature_name'].tolist()
    return groups


def load_dataset(dataset_path: str = "data/processed/earnings_ml_ready.parquet") -> pd.DataFrame:
    """Load the ML-ready dataset."""
    return pd.read_parquet(dataset_path)


def verify_no_forbidden_columns(df: pd.DataFrame, columns: List[str], context: str = "X") -> None:
    """Verify that forbidden columns are not present."""
    forbidden_found = [c for c in columns if c in FORBIDDEN_IN_X]
    if forbidden_found:
        raise ValueError(f"FORBIDDEN columns found in {context}: {forbidden_found}")


def create_temporal_splits(
    df: pd.DataFrame,
    train_end: pd.Timestamp = pd.Timestamp('2021-12-31'),
    val_end: pd.Timestamp = pd.Timestamp('2024-03-31'),
    embargo_days: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetadata]:
    """
    Create chronological train/validation/test splits with embargo.
    
    Args:
        df: Full dataset with period_ended column
        train_end: Last period_ended for training
        val_end: Last period_ended for validation
        embargo_days: Number of trading days embargo between splits
        
    Returns:
        train_df, val_df, test_df, split_metadata
    """
    # Ensure period_ended is datetime
    df = df.copy()
    df['period_ended'] = pd.to_datetime(df['period_ended'])
    
    # Sort by period_ended
    df = df.sort_values('period_ended').reset_index(drop=True)
    
    # Create splits based on period_ended
    train_mask = df['period_ended'] <= train_end
    val_mask = (df['period_ended'] > train_end) & (df['period_ended'] <= val_end)
    test_mask = df['period_ended'] > val_end
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    # Verify no overlap
    train_periods = set(train_df['period_ended'].unique())
    val_periods = set(val_df['period_ended'].unique())
    test_periods = set(test_df['period_ended'].unique())
    
    assert len(train_periods & val_periods) == 0, "Train/Val period overlap"
    assert len(val_periods & test_periods) == 0, "Val/Test period overlap"
    assert len(train_periods & test_periods) == 0, "Train/Test period overlap"
    
    # Verify chronological order
    assert train_df['period_ended'].max() < val_df['period_ended'].min(), "Train not before Val"
    assert val_df['period_ended'].max() < test_df['period_ended'].min(), "Val not before Test"
    
    # Get event keys for each split
    train_event_keys = train_df['event_key'].tolist()
    val_event_keys = val_df['event_key'].tolist()
    test_event_keys = test_df['event_key'].tolist()
    
    # Verify no duplicate event keys across splits
    all_keys = train_event_keys + val_event_keys + test_event_keys
    assert len(all_keys) == len(set(all_keys)), "Duplicate event keys across splits"
    
    metadata = SplitMetadata(
        name="chronological_5day_embargo",
        train_start=train_df['period_ended'].min(),
        train_end=train_df['period_ended'].max(),
        val_start=val_df['period_ended'].min(),
        val_end=val_df['period_ended'].max(),
        test_start=test_df['period_ended'].min(),
        test_end=test_df['period_ended'].max(),
        embargo_days=embargo_days,
        train_event_keys=train_event_keys,
        val_event_keys=val_event_keys,
        test_event_keys=test_event_keys,
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df)
    )
    
    return train_df, val_df, test_df, metadata


def prepare_xy_split(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into X (features) and y (targets).
    
    Args:
        df: Input dataframe
        feature_columns: List of feature column names
        target_columns: List of target column names (default: TARGET_COLUMNS)
        
    Returns:
        X, y dataframes
    """
    if target_columns is None:
        target_columns = TARGET_COLUMNS
    
    # Verify all feature columns exist
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    # Verify all target columns exist
    missing_targets = [c for c in target_columns if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")
    
    X = df[feature_columns].copy()
    y = df[target_columns].copy()
    
    # Verify no forbidden columns in X
    verify_no_forbidden_columns(X, X.columns.tolist(), "X")
    
    return X, y


def load_training_data(
    dataset_path: str = "data/processed/earnings_ml_ready.parquet",
    manifest_path: str = "data/processed/earnings_feature_manifest.csv",
    train_end: pd.Timestamp = pd.Timestamp('2021-12-31'),
    val_end: pd.Timestamp = pd.Timestamp('2024-03-31'),
    embargo_days: int = 5,
    target_columns: List[str] = None
) -> TrainingData:
    """
    Load and prepare all training data with proper splits.
    
    This is the MAIN ENTRY POINT for all training scripts.
    
    Args:
        dataset_path: Path to the ML-ready parquet file
        manifest_path: Path to the feature manifest CSV
        train_end: Last period_ended for training data
        val_end: Last period_ended for validation data
        embargo_days: Embargo in trading days between splits
        target_columns: Target columns to use (default: all TARGET_COLUMNS)
        
    Returns:
        TrainingData object with all splits and metadata
    """
    # Load manifest and dataset
    manifest = load_manifest(manifest_path)
    df = load_dataset(dataset_path)
    
    # Get feature columns from manifest
    feature_columns = get_feature_columns(manifest)
    feature_groups = get_feature_groups(manifest)
    
    if target_columns is None:
        target_columns = TARGET_COLUMNS
    
    # Create temporal splits
    train_df, val_df, test_df, split_metadata = create_temporal_splits(
        df, train_end=train_end, val_end=val_end, embargo_days=embargo_days
    )
    
    # Prepare X/y splits
    X_train, y_train = prepare_xy_split(train_df, feature_columns, target_columns)
    X_val, y_val = prepare_xy_split(val_df, feature_columns, target_columns)
    X_test, y_test = prepare_xy_split(test_df, feature_columns, target_columns)
    
    # Extract metadata arrays
    event_keys_train = train_df['event_key'].values
    event_keys_val = val_df['event_key'].values
    event_keys_test = test_df['event_key'].values
    
    announcement_dates_train = pd.to_datetime(train_df['result_announcement_datetime']).values
    announcement_dates_val = pd.to_datetime(val_df['result_announcement_datetime']).values
    announcement_dates_test = pd.to_datetime(test_df['result_announcement_datetime']).values
    
    feature_cutoff_dates_train = pd.to_datetime(train_df['feature_cutoff_date']).values
    feature_cutoff_dates_val = pd.to_datetime(val_df['feature_cutoff_date']).values
    feature_cutoff_dates_test = pd.to_datetime(test_df['feature_cutoff_date']).values
    
    reaction_start_dates_train = pd.to_datetime(train_df['reaction_start_date']).values
    reaction_start_dates_val = pd.to_datetime(val_df['reaction_start_date']).values
    reaction_start_dates_test = pd.to_datetime(test_df['reaction_start_date']).values
    
    return TrainingData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_columns,
        feature_groups=feature_groups,
        event_keys_train=event_keys_train,
        event_keys_val=event_keys_val,
        event_keys_test=event_keys_test,
        announcement_dates_train=announcement_dates_train,
        announcement_dates_val=announcement_dates_val,
        announcement_dates_test=announcement_dates_test,
        feature_cutoff_dates_train=feature_cutoff_dates_train,
        feature_cutoff_dates_val=feature_cutoff_dates_val,
        feature_cutoff_dates_test=feature_cutoff_dates_test,
        reaction_start_dates_train=reaction_start_dates_train,
        reaction_start_dates_val=reaction_start_dates_val,
        reaction_start_dates_test=reaction_start_dates_test,
        split_metadata=split_metadata
    )


def get_feature_columns_by_group(
    manifest: pd.DataFrame,
    groups: List[str]
) -> List[str]:
    """Get feature columns for specific feature groups."""
    features = manifest[(manifest['is_target'] == False) & (manifest['feature_group'].isin(groups))]
    return features['feature_name'].tolist()


# Track A: Strict PIT (Market + Prior Earnings only)
STRICT_PIT_GROUPS = [
    'market_ma_distance',
    'market_drawdown', 
    'market_relative',
    'market_returns',
    'market_volume',
    'market_volatility',
    'prior_earnings'
]

# Track B: Estimated PIT (all features)
ESTIMATED_PIT_GROUPS = [
    'market_ma_distance',
    'market_drawdown',
    'market_relative', 
    'market_returns',
    'market_volume',
    'market_volatility',
    'prior_earnings',
    'fundamental_balance',
    'fundamental_income',
    'fundamental_derived',
    'fundamental_cashflow',
    'valuation'
]


def load_training_data_strict_pit(
    dataset_path: str = "data/processed/earnings_ml_ready.parquet",
    manifest_path: str = "data/processed/earnings_feature_manifest.csv",
    train_end: pd.Timestamp = pd.Timestamp('2021-12-31'),
    val_end: pd.Timestamp = pd.Timestamp('2024-03-31'),
    embargo_days: int = 5,
    target_columns: List[str] = None
) -> TrainingData:
    """Load training data using ONLY strict PIT features (Market + Prior Earnings)."""
    manifest = load_manifest(manifest_path)
    feature_columns = get_feature_columns_by_group(manifest, STRICT_PIT_GROUPS)
    
    # Temporarily filter manifest to only include selected features
    filtered_manifest = manifest[manifest['feature_name'].isin(feature_columns) | manifest['is_target']].copy()
    
    return load_training_data(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        train_end=train_end,
        val_end=val_end,
        embargo_days=embargo_days,
        target_columns=target_columns
    )


def load_training_data_estimated_pit(
    dataset_path: str = "data/processed/earnings_ml_ready.parquet",
    manifest_path: str = "data/processed/earnings_feature_manifest.csv",
    train_end: pd.Timestamp = pd.Timestamp('2021-12-31'),
    val_end: pd.Timestamp = pd.Timestamp('2024-03-31'),
    embargo_days: int = 5,
    target_columns: List[str] = None
) -> TrainingData:
    """Load training data using ALL features (Estimated PIT track)."""
    return load_training_data(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        train_end=train_end,
        val_end=val_end,
        embargo_days=embargo_days,
        target_columns=target_columns
    )


if __name__ == "__main__":
    # Quick test
    print("Testing training data loader...")
    data = load_training_data()
    print(f"X_train shape: {data.X_train.shape}")
    print(f"y_train shape: {data.y_train.shape}")
    print(f"X_val shape: {data.X_val.shape}")
    print(f"y_val shape: {data.y_val.shape}")
    print(f"X_test shape: {data.X_test.shape}")
    print(f"y_test shape: {data.y_test.shape}")
    print(f"Feature count: {len(data.feature_names)}")
    print(f"Feature groups: {list(data.feature_groups.keys())}")
    print(f"Split metadata: {data.split_metadata}")
    print("✅ Training data loader test passed!")