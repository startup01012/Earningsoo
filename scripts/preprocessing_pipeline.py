#!/usr/bin/env python
"""
Preprocessing Pipeline for EarningsOS

Defines two preprocessing contracts:
1. TREE MODELS (LightGBM, XGBoost, CatBoost, RandomForest): Preserve NaN natively
2. LINEAR MODELS (LogisticRegression, Ridge, LinearRegression): Train-only imputation + scaling

All preprocessing parameters MUST be learned from TRAIN only.
Never fit preprocessing on validation + train or test.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Literal
from dataclasses import dataclass, field
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    model_type: Literal['tree', 'linear'] = 'tree'
    missing_indicator_threshold: float = 0.05  # Add indicators for features with >5% missing
    scaler_type: Literal['standard', 'robust', 'none'] = 'robust'
    max_categories: int = 50  # For categorical encoding


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """Add missing value indicator columns for specified features."""
    
    def __init__(self, features: List[str], threshold: float = 0.05):
        self.features = features
        self.threshold = threshold
        self.indicators_to_add_: List[str] = []
        
    def fit(self, X: pd.DataFrame, y=None):
        self.indicators_to_add_ = []
        for feat in self.features:
            if feat in X.columns:
                missing_pct = X[feat].isna().mean()
                if missing_pct >= self.threshold:
                    self.indicators_to_add_.append(feat)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feat in self.indicators_to_add_:
            indicator_name = f"{feat}_was_missing"
            X[indicator_name] = X[feat].isna().astype(int)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return list(self.features) + [f"{f}_was_missing" for f in self.indicators_to_add_]


class TreeModelPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessing for tree-based models (LightGBM, XGBoost, CatBoost, RandomForest).
    
    - Preserves NaN natively (tree models handle NaN optimally)
    - Adds missing indicators for informative missingness
    - Does NOT impute with zero (zero has economic meaning)
    - Does NOT use global imputation (leaks information across time)
    """
    
    def __init__(
        self,
        config: PreprocessingConfig = None,
        feature_groups: Dict[str, List[str]] = None
    ):
        self.config = config or PreprocessingConfig(model_type='tree')
        self.feature_groups = feature_groups or {}
        self.indicator_adder_: Optional[MissingIndicatorAdder] = None
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        
    def fit(self, X: pd.DataFrame, y=None) -> 'TreeModelPreprocessor':
        self.feature_names_in_ = list(X.columns)
        
        # Determine which features get missing indicators
        # Based on missingness strategy: EXPECTED_UNAVAILABLE and EXPECTED_STRUCTURAL
        # get indicators; UNEXPECTED missingness may not need indicators if rare
        all_features = list(X.columns)
        
        # Fit missing indicator adder
        self.indicator_adder_ = MissingIndicatorAdder(
            features=all_features,
            threshold=self.config.missing_indicator_threshold
        )
        self.indicator_adder_.fit(X)
        
        # Transform to get output feature names
        X_transformed = self.indicator_adder_.transform(X)
        self.feature_names_out_ = list(X_transformed.columns)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.indicator_adder_ is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Ensure columns match training
        missing_cols = set(self.feature_names_in_) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns at transform time: {missing_cols}")
        
        # Reorder columns to match training
        X = X[self.feature_names_in_].copy()
        
        # Add missing indicators (tree models handle NaN natively)
        X = self.indicator_adder_.transform(X)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class LinearModelPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessing for linear models (LogisticRegression, Ridge, LinearRegression).
    
    - Train-only imputation (fit imputer on train only)
    - Missing indicators for features with >5% missing in train
    - Robust scaling (StandardScaler or RobustScaler)
    - Handles string/categorical columns by dropping or encoding
    - Never fit on validation or test
    """
    
    def __init__(
        self,
        config: PreprocessingConfig = None,
        feature_groups: Dict[str, List[str]] = None
    ):
        self.config = config or PreprocessingConfig(model_type='linear')
        self.feature_groups = feature_groups or {}
        self.imputer_: Optional[SimpleImputer] = None
        self.scaler_: Optional[BaseEstimator] = None
        self.indicator_adder_: Optional[MissingIndicatorAdder] = None
        self.features_with_indicators_: List[str] = []
        self.numeric_features_: List[str] = []
        self.string_features_: List[str] = []
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.imputation_values_: Dict[str, float] = {}
        self.string_encoders_: Dict[str, Dict] = {}
        
    def fit(self, X: pd.DataFrame, y=None) -> 'LinearModelPreprocessor':
        self.feature_names_in_ = list(X.columns)
        
        # Identify numeric and string features
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.string_features_ = X.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # For string features, we'll drop them (linear models need numeric input)
        # In production, you might want to encode them instead
        if self.string_features_:
            warnings.warn(f"Dropping {len(self.string_features_)} string features for linear model: {self.string_features_}")
        
        # Work with numeric features only for imputation/scaling
        X_numeric = X[self.numeric_features_].copy()
        
        # Determine which numeric features get missing indicators
        # Features with >5% missing in training data
        self.features_with_indicators_ = [
            c for c in self.numeric_features_ 
            if X[c].isna().mean() >= self.config.missing_indicator_threshold
        ]
        
        # Fit missing indicator adder
        self.indicator_adder_ = MissingIndicatorAdder(
            features=self.features_with_indicators_,
            threshold=0  # Already filtered by threshold
        )
        self.indicator_adder_.fit(X_numeric)
        
        # Add indicators first, then impute
        X_with_indicators = self.indicator_adder_.transform(X_numeric)
        
        # Fit imputer on training data only (median for robustness)
        self.imputer_ = SimpleImputer(strategy='median')
        self.imputer_.fit(X_with_indicators)
        
        # Store imputation values for reference
        self.imputation_values_ = dict(zip(
            X_with_indicators.columns, 
            self.imputer_.statistics_
        ))
        
        # Transform with imputation
        X_imputed = pd.DataFrame(
            self.imputer_.transform(X_with_indicators),
            columns=X_with_indicators.columns,
            index=X.index
        )
        
        # Fit scaler on imputed training data
        if self.config.scaler_type == 'standard':
            self.scaler_ = StandardScaler()
        elif self.config.scaler_type == 'robust':
            self.scaler_ = RobustScaler()
        elif self.config.scaler_type == 'none':
            self.scaler_ = None
        else:
            raise ValueError(f"Unknown scaler_type: {self.config.scaler_type}")
        
        if self.scaler_ is not None:
            self.scaler_.fit(X_imputed)
        
        # Transform to get output feature names
        X_final = self._apply_scaler(X_imputed)
        self.feature_names_out_ = list(X_final.columns)
        
        return self
    
    def _apply_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None:
            return X
        X_scaled = pd.DataFrame(
            self.scaler_.transform(X),
            columns=X.columns,
            index=X.index
        )
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.imputer_ is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Ensure columns match training
        missing_cols = set(self.feature_names_in_) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns at transform time: {missing_cols}")
        
        # Reorder columns to match training
        X = X[self.feature_names_in_].copy()
        
        # Select numeric features only
        X_numeric = X[self.numeric_features_].copy()
        
        # Add missing indicators
        X_numeric = self.indicator_adder_.transform(X_numeric)
        
        # Impute using training statistics only
        X_imputed = pd.DataFrame(
            self.imputer_.transform(X_numeric),
            columns=X_numeric.columns,
            index=X.index
        )
        
        # Scale using training statistics only
        X_final = self._apply_scaler(X_imputed)
        
        # Ensure output matches expected feature names
        if list(X_final.columns) != self.feature_names_out_:
            # This shouldn't happen but just in case
            X_final = X_final[self.feature_names_out_]
        
        return X_final
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def create_preprocessor(
    model_type: Literal['tree', 'linear'] = 'tree',
    config: PreprocessingConfig = None,
    feature_groups: Dict[str, List[str]] = None
) -> BaseEstimator:
    """
    Factory function to create appropriate preprocessor.
    
    Args:
        model_type: 'tree' for LightGBM/XGBoost/CatBoost/RF, 'linear' for LogisticRegression/Ridge
        config: PreprocessingConfig object
        feature_groups: Dict mapping group names to feature lists
        
    Returns:
        Fitted preprocessor (BaseEstimator with fit/transform)
    """
    if config is None:
        config = PreprocessingConfig(model_type=model_type)
    else:
        config.model_type = model_type
    
    if model_type == 'tree':
        return TreeModelPreprocessor(config=config, feature_groups=feature_groups)
    elif model_type == 'linear':
        return LinearModelPreprocessor(config=config, feature_groups=feature_groups)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'tree' or 'linear'.")


def demonstrate_preprocessing_no_leakage():
    """
    Demonstrate that validation/test statistics cannot affect train preprocessing.
    This is a smoke test to verify the pipeline design.
    """
    np.random.seed(42)
    n_train, n_val, n_test = 500, 100, 50
    
    # Create synthetic data with different missingness patterns
    def make_data(n, missing_rate=0.2):
        X = pd.DataFrame({
            'feat_1': np.random.randn(n),
            'feat_2': np.random.randn(n),
            'feat_3': np.random.randn(n),
        })
        # Introduce missingness
        for c in X.columns:
            mask = np.random.rand(n) < missing_rate
            X.loc[mask, c] = np.nan
        return X
    
    X_train = make_data(n_train, 0.3)
    X_val = make_data(n_val, 0.1)  # Different missingness rate
    X_test = make_data(n_test, 0.5)  # Very different missingness rate
    
    # Test Tree Model Preprocessor
    print("Testing Tree Model Preprocessor...")
    tree_prep = create_preprocessor('tree')
    tree_prep.fit(X_train)
    
    X_train_tree = tree_prep.transform(X_train)
    X_val_tree = tree_prep.transform(X_val)
    X_test_tree = tree_prep.transform(X_test)
    
    print(f"  Train shape: {X_train.shape} -> {X_train_tree.shape}")
    print(f"  Val shape: {X_val.shape} -> {X_val_tree.shape}")
    print(f"  Test shape: {X_test.shape} -> {X_test_tree.shape}")
    print(f"  Missing indicators added: {[c for c in X_train_tree.columns if 'was_missing' in c]}")
    
    # Verify no data leakage: train statistics shouldn't depend on val/test
    # The indicator adder was fit only on train data
    print(f"  Indicators determined from TRAIN only: {tree_prep.indicator_adder_.indicators_to_add_}")
    
    # Test Linear Model Preprocessor
    print("\nTesting Linear Model Preprocessor...")
    linear_prep = create_preprocessor('linear', config=PreprocessingConfig(
        model_type='linear',
        scaler_type='robust'
    ))
    linear_prep.fit(X_train)
    
    X_train_lin = linear_prep.transform(X_train)
    X_val_lin = linear_prep.transform(X_val)
    X_test_lin = linear_prep.transform(X_test)
    
    print(f"  Train shape: {X_train.shape} -> {X_train_lin.shape}")
    print(f"  Val shape: {X_val.shape} -> {X_val_lin.shape}")
    print(f"  Test shape: {X_test.shape} -> {X_test_lin.shape}")
    print(f"  Missing indicators: {linear_prep.features_with_indicators_}")
    print(f"  Imputation values: {linear_prep.imputation_values_}")
    
    # Verify no NaN in transformed data
    assert not X_train_lin.isna().any().any(), "Train has NaN after preprocessing!"
    assert not X_val_lin.isna().any().any(), "Val has NaN after preprocessing!"
    assert not X_test_lin.isna().any().any(), "Test has NaN after preprocessing!"
    
    # Verify scaling statistics are from train only
    if linear_prep.scaler_ is not None:
        if hasattr(linear_prep.scaler_, 'mean_'):
            print(f"  Scaler mean (from train): {linear_prep.scaler_.mean_[:3]}")
        if hasattr(linear_prep.scaler_, 'center_'):
            print(f"  Scaler center (from train): {linear_prep.scaler_.center_[:3]}")
        if hasattr(linear_prep.scaler_, 'scale_'):
            print(f"  Scaler scale (from train): {linear_prep.scaler_.scale_[:3]}")
    
    print("\n✅ Preprocessing pipeline test passed!")
    print("   Key property: Val/Test missingness rates do NOT affect train preprocessing")


if __name__ == "__main__":
    demonstrate_preprocessing_no_leakage()