#!/usr/bin/env python
"""
Tests for the Training-Time Leakage Firewall
"""

import numpy as np
import pandas as pd
import pytest
from training_leakage_firewall import LeakageFirewall, LeakageFirewallError


class TestLeakageFirewall:
    """Tests for the LeakageFirewall class."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 100
        
        # Create synthetic feature data (no forbidden columns)
        self.X_train = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'prior_earnings_count': np.arange(n),
        })
        self.X_val = pd.DataFrame({
            'feature_1': np.random.randn(30),
            'feature_2': np.random.randn(30),
            'prior_earnings_count': np.arange(30),
        })
        self.X_test = pd.DataFrame({
            'feature_1': np.random.randn(20),
            'feature_2': np.random.randn(20),
            'prior_earnings_count': np.arange(20),
        })
        
        # Valid dates - chronological (train < val < test)
        # Train: 2018-01-01 to 2020-01-01 (100 weeks)
        base = pd.Timestamp('2018-01-01')
        self.train_fc = pd.date_range(base, periods=n, freq='W')
        self.train_rs = self.train_fc + pd.Timedelta(days=10)
        self.train_ann = self.train_fc + pd.Timedelta(days=5)
        self.train_pe = self.train_fc
        
        # Val: 2020-01-15 to 2020-07-15 (30 weeks) - AFTER train
        val_start = self.train_fc[-1] + pd.Timedelta(weeks=2)
        self.val_fc = pd.date_range(val_start, periods=30, freq='W')
        self.val_rs = self.val_fc + pd.Timedelta(days=10)
        self.val_ann = self.val_fc + pd.Timedelta(days=5)
        self.val_pe = self.val_fc
        
        # Test: 2020-08-01 to 2020-12-01 (20 weeks) - AFTER val
        test_start = self.val_fc[-1] + pd.Timedelta(weeks=2)
        self.test_fc = pd.date_range(test_start, periods=20, freq='W')
        self.test_rs = self.test_fc + pd.Timedelta(days=10)
        self.test_ann = self.test_fc + pd.Timedelta(days=5)
        self.test_pe = self.test_fc
        
        self.train_keys = np.array([f'train_{i}' for i in range(n)])
        self.val_keys = np.array([f'val_{i}' for i in range(30)])
        self.test_keys = np.array([f'test_{i}' for i in range(20)])
        
    def test_valid_data_passes(self):
        """Test that valid data passes all checks."""
        firewall = LeakageFirewall(strict=True)
        firewall.check_all(
            X_train=self.X_train, X_val=self.X_val, X_test=self.X_test,
            train_event_keys=self.train_keys, val_event_keys=self.val_keys, test_event_keys=self.test_keys,
            train_feature_cutoff_dates=self.train_fc, val_feature_cutoff_dates=self.val_fc, test_feature_cutoff_dates=self.test_fc,
            train_reaction_start_dates=self.train_rs, val_reaction_start_dates=self.val_rs, test_reaction_start_dates=self.test_rs,
            train_announcement_dates=self.train_ann, val_announcement_dates=self.val_ann, test_announcement_dates=self.test_ann,
            train_period_ends=self.train_pe, val_period_ends=self.val_pe, test_period_ends=self.test_pe
        )
        # All checks should pass
        assert all(r.passed for r in firewall.results)
        
    def test_target_leakage_fails(self):
        """Test that target columns in X cause failure."""
        X_bad = self.X_train.copy()
        X_bad['reaction_class'] = 'NEUTRAL'  # Forbidden!
        
        firewall = LeakageFirewall(strict=True)
        with pytest.raises(LeakageFirewallError, match="No target leakage"):
            firewall.check_no_target_leakage(X_bad, "X_train")
            
    def test_post_event_columns_fails(self):
        """Test that _after columns in X cause failure."""
        X_bad = self.X_train.copy()
        X_bad['return_1d_after'] = 0.01  # Forbidden!
        
        firewall = LeakageFirewall(strict=True)
        with pytest.raises(LeakageFirewallError, match="No post-event columns"):
            firewall.check_no_post_event_columns(X_bad, "X_train")
            
    def test_future_looking_names_fails(self):
        """Test that future-looking column names cause failure."""
        X_bad = self.X_train.copy()
        X_bad['future_return'] = 0.01  # Suspicious!
        
        firewall = LeakageFirewall(strict=True)
        with pytest.raises(LeakageFirewallError, match="future-looking"):
            firewall.check_no_future_looking_names(X_bad, "X_train")
            
    def test_feature_cutoff_after_reaction_fails(self):
        """Test that feature_cutoff >= reaction_start causes failure."""
        # Make feature_cutoff AFTER reaction_start (violation)
        bad_fc = self.train_rs + pd.Timedelta(days=1)
        
        firewall = LeakageFirewall(strict=True)
        with pytest.raises(LeakageFirewallError, match="Feature cutoff < reaction start"):
            firewall.check_feature_cutoff_before_reaction(bad_fc, self.train_rs, "train")
            
    def test_feature_cutoff_after_announcement_fails(self):
        """Test that feature_cutoff > announcement causes failure."""
        # Make feature_cutoff AFTER announcement (violation)
        bad_fc = self.train_ann + pd.Timedelta(days=1)
        
        firewall = LeakageFirewall(strict=True)
        with pytest.raises(LeakageFirewallError, match="Feature cutoff <= announcement"):
            firewall.check_feature_cutoff_before_announcement(bad_fc, self.train_ann, "train")
            
    def test_non_chronological_splits_fail(self):
        """Test that non-chronological splits cause failure."""
        # Make val dates BEFORE train dates
        bad_val_fc = self.train_fc - pd.Timedelta(days=100)
        
        firewall = LeakageFirewall(strict=True)
        with pytest.raises(LeakageFirewallError, match="Chronological split order"):
            firewall.check_chronological_order(self.train_fc, bad_val_fc, self.test_fc, "period_ended")
            
    def test_overlapping_event_keys_fail(self):
        """Test that overlapping event keys cause failure."""
        # Make val keys overlap with train keys
        bad_val_keys = np.array(['train_0', 'val_1', 'val_2'])
        
        firewall = LeakageFirewall(strict=True)
        with pytest.raises(LeakageFirewallError, match="Disjoint event keys"):
            firewall.check_disjoint_event_keys(self.train_keys, bad_val_keys, self.test_keys)
            
    def test_duplicate_event_keys_fail(self):
        """Test that duplicate event keys cause failure."""
        # Add duplicate to train keys
        bad_train_keys = np.append(self.train_keys, 'train_0')
        
        firewall = LeakageFirewall(strict=True)
        with pytest.raises(LeakageFirewallError, match="duplicate event keys"):
            firewall.check_no_duplicate_event_keys(bad_train_keys, self.val_keys, self.test_keys)
            
    def test_fundamental_pit_violation_fails(self):
        """Test that fundamental PIT violations cause failure."""
        X_bad = self.X_train.copy()
        # Add fundamental columns with violation
        X_bad['fundamental_estimated_available_date'] = self.train_rs + pd.Timedelta(days=10)  # After feature_cutoff!
        X_bad['fundamental_period_end_date'] = self.train_pe + pd.Timedelta(days=10)  # After event period!
        
        firewall = LeakageFirewall(strict=True)
        with pytest.raises(LeakageFirewallError, match="Fundamental PIT dates"):
            firewall.check_fundamental_pit_dates(X_bad, self.train_fc, self.train_pe, "train")
            
    def test_non_strict_mode_collects_failures(self):
        """Test that non-strict mode collects failures without raising."""
        X_bad = self.X_train.copy()
        X_bad['reaction_class'] = 'NEUTRAL'
        
        firewall = LeakageFirewall(strict=False)
        firewall.check_no_target_leakage(X_bad, "X_train")
        
        # Should have recorded the failure but not raised
        assert len(firewall.results) == 1
        assert not firewall.results[0].passed
        assert "reaction_class" in firewall.results[0].details
        
    def test_summary_output(self):
        """Test that summary output works."""
        firewall = LeakageFirewall(strict=False)
        firewall.check_no_target_leakage(self.X_train, "X_train")
        firewall.check_no_target_leakage(self.X_train, "X_train")  # Add another
        
        summary = firewall.summary()
        assert "LEAKAGE FIREWALL SUMMARY" in summary
        assert "PASS" in summary
        assert "Total:" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])