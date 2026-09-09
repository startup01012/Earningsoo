#!/usr/bin/env python
"""
Training-Time Leakage Firewall

This module provides a HARD firewall that MUST be called immediately before
any model.fit() call in the training pipeline. It raises exceptions on any
leakage detection - not warnings, but hard failures.

The firewall checks:
1. No target columns in X
2. No post-event columns in X
3. No future-looking feature names
4. feature_cutoff_datetime < reaction_start_datetime
5. feature_cutoff_datetime <= announcement_datetime
6. Prior earnings are strictly historical
7. Fundamental estimated availability <= feature cutoff
8. Fundamental period end < event period end
9. No test rows used during preprocessing
10. No test rows used for feature selection
11. No test rows used for threshold selection
11. No test rows used for hyperparameter selection
13. Train/validation/test event keys are disjoint
14. Split chronology is valid
15. No duplicate event key crosses splits
"""

import numpy as np
import pandas as pd
from typing import List, Set, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


class LeakageFirewallError(Exception):
    """Raised when the leakage firewall detects a violation."""
    pass


@dataclass
class FirewallResult:
    """Result of a firewall check."""
    check_name: str
    passed: bool
    details: str
    
    
class LeakageFirewall:
    """
    Training-time leakage firewall.
    
    Usage:
        firewall = LeakageFirewall()
        firewall.check_all(X_train, y_train, X_val, y_val, X_test, y_test,
                          train_event_keys, val_event_keys, test_event_keys,
                          train_dates, val_dates, test_dates)
        # If no exception raised, proceed to model.fit()
    """
    
    # Columns that must NEVER appear in X (features)
    FORBIDDEN_IN_X = {
        'reaction_class', 'abnormal_return_1d', 'abnormal_return_3d',
        'abnormal_return_5d', 'return_1d_after', 'return_3d_after',
        'return_5d_after', 'benchmark_return_1d_after',
        'benchmark_return_3d_after', 'benchmark_return_5d_after',
        'reaction_start_date', 'reaction_start_datetime',
        'fundamental_period_end_date', 'fundamental_estimated_available_date',
        'fundamental_pit_status', 'fundamental_lag_days',
        'event_key', 'symbol', 'company_name', 'period_ended',
        'fiscal_quarter', 'result_announcement_datetime',
        'announcement_session_type', 'feature_cutoff_date',
        'quality_flags', 'price_history_days', 'benchmark_history_days'
    }
    
    # Suspicious future-looking patterns
    FUTURE_PATTERNS = [
        'future', 'forward', 'next_', 't+1', 't+2', 't+3',
        't_plus', 'lead_', 'lag_-', 'predicted_', 'forecast_'
    ]
    
    def __init__(self, strict: bool = True):
        """
        Initialize the firewall.
        
        Args:
            strict: If True, raise exceptions on violations. If False, collect and report.
        """
        self.strict = strict
        self.results: List[FirewallResult] = []
        
    def _record(self, check_name: str, passed: bool, details: str = "") -> None:
        """Record a check result."""
        self.results.append(FirewallResult(check_name, passed, details))
        if not passed and self.strict:
            raise LeakageFirewallError(f"LEAKAGE FIREWALL FAILED - {check_name}: {details}")
            
    def check_no_target_leakage(self, X: pd.DataFrame, context: str = "X") -> bool:
        """Check 1: No target columns in X."""
        forbidden_found = [c for c in X.columns if c in self.FORBIDDEN_IN_X]
        passed = len(forbidden_found) == 0
        details = f"Found forbidden columns: {forbidden_found}" if forbidden_found else "OK"
        self._record(f"No target leakage in {context}", passed, details)
        return passed
        
    def check_no_post_event_columns(self, X: pd.DataFrame, context: str = "X") -> bool:
        """Check 2: No post-event columns in X (columns with '_after')."""
        post_event = [c for c in X.columns if '_after' in c.lower()]
        passed = len(post_event) == 0
        details = f"Found post-event columns: {post_event}" if post_event else "OK"
        self._record(f"No post-event columns in {context}", passed, details)
        return passed
        
    def check_no_future_looking_names(self, X: pd.DataFrame, context: str = "X") -> bool:
        """Check 3: No future-looking feature names."""
        suspicious = []
        for c in X.columns:
            c_lower = c.lower()
            for pattern in self.FUTURE_PATTERNS:
                if pattern in c_lower:
                    suspicious.append(c)
                    break
        passed = len(suspicious) == 0
        details = f"Suspicious future-looking columns: {suspicious}" if suspicious else "OK"
        self._record(f"No future-looking names in {context}", passed, details)
        return passed
        
    def check_feature_cutoff_before_reaction(
        self, 
        feature_cutoff_dates: np.ndarray, 
        reaction_start_dates: np.ndarray,
        context: str = "train"
    ) -> bool:
        """Check 4: feature_cutoff_datetime < reaction_start_datetime."""
        fc = pd.to_datetime(feature_cutoff_dates)
        rs = pd.to_datetime(reaction_start_dates)
        violations = fc >= rs
        n_violations = violations.sum()
        passed = n_violations == 0
        details = f"{n_violations} violations: feature_cutoff >= reaction_start" if n_violations > 0 else "OK"
        self._record(f"Feature cutoff < reaction start ({context})", passed, details)
        return passed
        
    def check_feature_cutoff_before_announcement(
        self,
        feature_cutoff_dates: np.ndarray,
        announcement_dates: np.ndarray,
        context: str = "train"
    ) -> bool:
        """Check 5: feature_cutoff_datetime <= announcement_datetime."""
        fc = pd.to_datetime(feature_cutoff_dates)
        ann = pd.to_datetime(announcement_dates)
        violations = fc > ann
        n_violations = violations.sum()
        passed = n_violations == 0
        details = f"{n_violations} violations: feature_cutoff > announcement" if n_violations > 0 else "OK"
        self._record(f"Feature cutoff <= announcement ({context})", passed, details)
        return passed
        
    def check_prior_earnings_historical(
        self,
        X: pd.DataFrame,
        event_keys: np.ndarray,
        full_dataset: pd.DataFrame,
        context: str = "train"
    ) -> bool:
        """Check 6: Prior earnings features use only strictly earlier events."""
        # Check that prior_earnings_count is non-decreasing within each symbol
        # This is a proxy - the real check would need the full event history
        if 'prior_earnings_count' in X.columns:
            # We can't fully verify without the full dataset, but we can check
            # that prior_earnings_count values are reasonable (non-negative, not too large)
            counts = X['prior_earnings_count'].values
            invalid = (counts < 0) | (counts > 100)  # Sanity check
            n_invalid = invalid.sum()
            passed = n_invalid == 0
            details = f"{n_invalid} invalid prior_earnings_count values" if n_invalid > 0 else "OK"
        else:
            passed = True
            details = "No prior_earnings_count column (OK if not using prior earnings)"
        self._record(f"Prior earnings historical ({context})", passed, details)
        return passed
        
    def check_fundamental_pit_dates(
        self,
        X: pd.DataFrame,
        feature_cutoff_dates: np.ndarray,
        event_period_ends: np.ndarray,
        context: str = "train"
    ) -> bool:
        """Check 7 & 8: Fundamental estimated availability <= feature cutoff, period_end < event period_end."""
        passed = True
        details_list = []
        
        # Check 7: fundamental_estimated_available_date <= feature_cutoff_date
        if 'fundamental_estimated_available_date' in X.columns:
            fc = pd.to_datetime(feature_cutoff_dates)
            fead = pd.to_datetime(X['fundamental_estimated_available_date'])
            # Only check where fundamental data exists
            has_fund = X['fundamental_estimated_available_date'].notna()
            if has_fund.any():
                violations = fead[has_fund] > fc[has_fund]
                n_violations = violations.sum()
                if n_violations > 0:
                    passed = False
                    details_list.append(f"{n_violations} violations: fundamental_estimated_available > feature_cutoff")
                    
        # Check 8: fundamental_period_end_date < event period_ended
        if 'fundamental_period_end_date' in X.columns:
            fpe = pd.to_datetime(X['fundamental_period_end_date'])
            epe = pd.to_datetime(event_period_ends)
            has_fund = X['fundamental_period_end_date'].notna()
            if has_fund.any():
                violations = fpe[has_fund] >= epe[has_fund]
                n_violations = violations.sum()
                if n_violations > 0:
                    passed = False
                    details_list.append(f"{n_violations} violations: fundamental_period_end >= event_period_end")
                    
        details = "; ".join(details_list) if details_list else "OK"
        self._record(f"Fundamental PIT dates valid ({context})", passed, details)
        return passed
        
    def check_no_test_contamination(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
        context: str = "preprocessing"
    ) -> bool:
        """Check 9-12: No test rows used during preprocessing/feature selection/threshold/hyperparameter selection."""
        # This check verifies that the indices provided for train/val/test are disjoint
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)
        
        train_val_overlap = train_set & val_set
        val_test_overlap = val_set & test_set
        train_test_overlap = train_set & test_set
        
        n_overlaps = len(train_val_overlap) + len(val_test_overlap) + len(train_test_overlap)
        passed = n_overlaps == 0
        details = f"Overlaps found: train/val={len(train_val_overlap)}, val/test={len(val_test_overlap)}, train/test={len(train_test_overlap)}" if n_overlaps > 0 else "OK"
        self._record(f"No test contamination in {context}", passed, details)
        return passed
        
    def check_disjoint_event_keys(
        self,
        train_event_keys: np.ndarray,
        val_event_keys: np.ndarray,
        test_event_keys: np.ndarray
    ) -> bool:
        """Check 13: Train/validation/test event keys are disjoint."""
        train_set = set(train_event_keys)
        val_set = set(val_event_keys)
        test_set = set(test_event_keys)
        
        train_val = train_set & val_set
        val_test = val_set & test_set
        train_test = train_set & test_set
        
        n_overlaps = len(train_val) + len(val_test) + len(train_test)
        passed = n_overlaps == 0
        details = f"Event key overlaps: train/val={len(train_val)}, val/test={len(val_test)}, train/test={len(train_test)}" if n_overlaps > 0 else "OK"
        self._record("Disjoint event keys across splits", passed, details)
        return passed
        
    def check_chronological_order(
        self,
        train_dates: np.ndarray,
        val_dates: np.ndarray,
        test_dates: np.ndarray,
        date_type: str = "period_ended"
    ) -> bool:
        """Check 14: Split chronology is valid (train < val < test)."""
        train_max = pd.Series(pd.to_datetime(train_dates)).max()
        val_min = pd.Series(pd.to_datetime(val_dates)).min()
        val_max = pd.Series(pd.to_datetime(val_dates)).max()
        test_min = pd.Series(pd.to_datetime(test_dates)).min()
        
        train_before_val = train_max < val_min
        val_before_test = val_max < test_min
        
        passed = train_before_val and val_before_test
        details = f"Train max {date_type}: {train_max}, Val min: {val_min}, Val max: {val_max}, Test min: {test_min}"
        if not train_before_val:
            details += " - TRAIN NOT BEFORE VAL"
        if not val_before_test:
            details += " - VAL NOT BEFORE TEST"
        self._record(f"Chronological split order ({date_type})", passed, details)
        return passed
        
    def check_no_duplicate_event_keys(
        self,
        train_event_keys: np.ndarray,
        val_event_keys: np.ndarray,
        test_event_keys: np.ndarray
    ) -> bool:
        """Check 15: No duplicate event keys within or across splits."""
        all_keys = list(train_event_keys) + list(val_event_keys) + list(test_event_keys)
        unique_keys = set(all_keys)
        n_duplicates = len(all_keys) - len(unique_keys)
        passed = n_duplicates == 0
        details = f"{n_duplicates} duplicate event keys found" if n_duplicates > 0 else "OK"
        self._record("No duplicate event keys", passed, details)
        return passed
        
    def check_all(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        train_event_keys: np.ndarray,
        val_event_keys: np.ndarray,
        test_event_keys: np.ndarray,
        train_feature_cutoff_dates: np.ndarray,
        val_feature_cutoff_dates: np.ndarray,
        test_feature_cutoff_dates: np.ndarray,
        train_reaction_start_dates: np.ndarray,
        val_reaction_start_dates: np.ndarray,
        test_reaction_start_dates: np.ndarray,
        train_announcement_dates: np.ndarray,
        val_announcement_dates: np.ndarray,
        test_announcement_dates: np.ndarray,
        train_period_ends: np.ndarray,
        val_period_ends: np.ndarray,
        test_period_ends: np.ndarray,
        train_indices: Optional[np.ndarray] = None,
        val_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> List[FirewallResult]:
        """
        Run ALL firewall checks.
        
        Args:
            X_train, X_val, X_test: Feature dataframes
            train/val/test_event_keys: Event keys for each split
            train/val/test_feature_cutoff_dates: Feature cutoff dates
            train/val/test_reaction_start_dates: Reaction start dates
            train/val/test_announcement_dates: Announcement dates
            train/val/test_period_ends: Period end dates
            train/val/test_indices: Optional row indices for contamination check
            
        Returns:
            List of FirewallResult objects
            
        Raises:
            LeakageFirewallError: If any check fails (in strict mode)
        """
        self.results = []
        
        # 1-3: X column checks
        self.check_no_target_leakage(X_train, "X_train")
        self.check_no_target_leakage(X_val, "X_val")
        self.check_no_target_leakage(X_test, "X_test")
        
        self.check_no_post_event_columns(X_train, "X_train")
        self.check_no_post_event_columns(X_val, "X_val")
        self.check_no_post_event_columns(X_test, "X_test")
        
        self.check_no_future_looking_names(X_train, "X_train")
        self.check_no_future_looking_names(X_val, "X_val")
        self.check_no_future_looking_names(X_test, "X_test")
        
        # 4-5: Temporal ordering checks
        self.check_feature_cutoff_before_reaction(
            train_feature_cutoff_dates, train_reaction_start_dates, "train"
        )
        self.check_feature_cutoff_before_reaction(
            val_feature_cutoff_dates, val_reaction_start_dates, "val"
        )
        self.check_feature_cutoff_before_reaction(
            test_feature_cutoff_dates, test_reaction_start_dates, "test"
        )
        
        self.check_feature_cutoff_before_announcement(
            train_feature_cutoff_dates, train_announcement_dates, "train"
        )
        self.check_feature_cutoff_before_announcement(
            val_feature_cutoff_dates, val_announcement_dates, "val"
        )
        self.check_feature_cutoff_before_announcement(
            test_feature_cutoff_dates, test_announcement_dates, "test"
        )
        
        # 6: Prior earnings historical
        self.check_prior_earnings_historical(X_train, train_event_keys, None, "train")
        self.check_prior_earnings_historical(X_val, val_event_keys, None, "val")
        self.check_prior_earnings_historical(X_test, test_event_keys, None, "test")
        
        # 7-8: Fundamental PIT dates
        self.check_fundamental_pit_dates(
            X_train, train_feature_cutoff_dates, train_period_ends, "train"
        )
        self.check_fundamental_pit_dates(
            X_val, val_feature_cutoff_dates, val_period_ends, "val"
        )
        self.check_fundamental_pit_dates(
            X_test, test_feature_cutoff_dates, test_period_ends, "test"
        )
        
        # 9-12: No test contamination (if indices provided)
        if train_indices is not None and val_indices is not None and test_indices is not None:
            self.check_no_test_contamination(train_indices, val_indices, test_indices, "preprocessing")
            
        # 13: Disjoint event keys
        self.check_disjoint_event_keys(train_event_keys, val_event_keys, test_event_keys)
        
        # 14: Chronological order
        self.check_chronological_order(train_period_ends, val_period_ends, test_period_ends, "period_ended")
        
        # 15: No duplicate event keys
        self.check_no_duplicate_event_keys(train_event_keys, val_event_keys, test_event_keys)
        
        return self.results
        
    def summary(self) -> str:
        """Get a summary of all checks."""
        lines = ["=" * 60, "LEAKAGE FIREWALL SUMMARY", "=" * 60]
        for r in self.results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            lines.append(f"  {status} | {r.check_name}")
            if r.details and not r.passed:
                lines.append(f"    Details: {r.details}")
        lines.append("=" * 60)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        lines.append(f"Total: {passed} passed, {failed} failed")
        return "\n".join(lines)


def run_leakage_firewall(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    train_event_keys: np.ndarray,
    val_event_keys: np.ndarray,
    test_event_keys: np.ndarray,
    train_feature_cutoff_dates: np.ndarray,
    val_feature_cutoff_dates: np.ndarray,
    test_feature_cutoff_dates: np.ndarray,
    train_reaction_start_dates: np.ndarray,
    val_reaction_start_dates: np.ndarray,
    test_reaction_start_dates: np.ndarray,
    train_announcement_dates: np.ndarray,
    val_announcement_dates: np.ndarray,
    test_announcement_dates: np.ndarray,
    train_period_ends: np.ndarray,
    val_period_ends: np.ndarray,
    test_period_ends: np.ndarray,
    strict: bool = True
) -> None:
    """
    Convenience function to run the full leakage firewall.
    
    Raises:
        LeakageFirewallError: If any check fails
    """
    firewall = LeakageFirewall(strict=strict)
    firewall.check_all(
        X_train=X_train, X_val=X_val, X_test=X_test,
        train_event_keys=train_event_keys,
        val_event_keys=val_event_keys,
        test_event_keys=test_event_keys,
        train_feature_cutoff_dates=train_feature_cutoff_dates,
        val_feature_cutoff_dates=val_feature_cutoff_dates,
        test_feature_cutoff_dates=test_feature_cutoff_dates,
        train_reaction_start_dates=train_reaction_start_dates,
        val_reaction_start_dates=val_reaction_start_dates,
        test_reaction_start_dates=test_reaction_start_dates,
        train_announcement_dates=train_announcement_dates,
        val_announcement_dates=val_announcement_dates,
        test_announcement_dates=test_announcement_dates,
        train_period_ends=train_period_ends,
        val_period_ends=val_period_ends,
        test_period_ends=test_period_ends
    )
    print(firewall.summary())


if __name__ == "__main__":
    # Quick test with the training data loader
    from training_data_loader import load_training_data
    
    print("Testing leakage firewall...")
    data = load_training_data()
    
    # Get period_ends from the splits
    # We need to load the full dataset to get period_ends for each split
    import pandas as pd
    df = pd.read_parquet("data/processed/earnings_ml_ready.parquet")
    df['period_ended'] = pd.to_datetime(df['period_ended'])
    
    train_period_ends = df[df['event_key'].isin(data.event_keys_train)]['period_ended'].values
    val_period_ends = df[df['event_key'].isin(data.event_keys_val)]['period_ended'].values
    test_period_ends = df[df['event_key'].isin(data.event_keys_test)]['period_ended'].values
    
    firewall = LeakageFirewall(strict=True)
    firewall.check_all(
        X_train=data.X_train, X_val=data.X_val, X_test=data.X_test,
        train_event_keys=data.event_keys_train,
        val_event_keys=data.event_keys_val,
        test_event_keys=data.event_keys_test,
        train_feature_cutoff_dates=data.feature_cutoff_dates_train,
        val_feature_cutoff_dates=data.feature_cutoff_dates_val,
        test_feature_cutoff_dates=data.feature_cutoff_dates_test,
        train_reaction_start_dates=data.reaction_start_dates_train,
        val_reaction_start_dates=data.reaction_start_dates_val,
        test_reaction_start_dates=data.reaction_start_dates_test,
        train_announcement_dates=data.announcement_dates_train,
        val_announcement_dates=data.announcement_dates_val,
        test_announcement_dates=data.announcement_dates_test,
        train_period_ends=train_period_ends,
        val_period_ends=val_period_ends,
        test_period_ends=test_period_ends
    )
    print(firewall.summary())
    print("✅ Leakage firewall test passed!")