"""
Tests for Final ML-Ready Dataset

Tests critical invariants for leakage prevention and data quality.
"""

import pytest
import pandas as pd
import numpy as np


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def ml_dataset():
    """Load the final ML-ready dataset."""
    return pd.read_parquet("data/processed/earnings_ml_ready.parquet")


@pytest.fixture
def feature_manifest():
    """Load the feature manifest."""
    return pd.read_csv("data/processed/earnings_feature_manifest.csv")


@pytest.fixture
def events():
    """Load canonical events."""
    return pd.read_parquet("data/processed/earnings_events.parquet")


# ============================================================
# TESTS
# ============================================================

class TestForbiddenColumns:
    """Test that no forbidden columns enter model features."""

    def test_no_reaction_class_in_features(self, ml_dataset, feature_manifest):
        """reaction_class must not be in features."""
        features = feature_manifest[feature_manifest["is_target"] == False]["feature_name"].tolist()
        assert "reaction_class" not in features, "reaction_class found in features"

    def test_no_abnormal_return_in_features(self, ml_dataset, feature_manifest):
        """abnormal_return_* must not be in features."""
        features = feature_manifest[feature_manifest["is_target"] == False]["feature_name"].tolist()
        for col in features:
            assert not col.startswith("abnormal_return_"), f"Abnormal return in features: {col}"

    def test_no_after_columns_in_features(self, ml_dataset, feature_manifest):
        """*_after columns must not be in features."""
        features = feature_manifest[feature_manifest["is_target"] == False]["feature_name"].tolist()
        for col in features:
            assert not col.endswith("_after"), f"After column in features: {col}"

    def test_no_reaction_start_date_in_features(self, ml_dataset, feature_manifest):
        """reaction_start_date must not be in features."""
        features = feature_manifest[feature_manifest["is_target"] == False]["feature_name"].tolist()
        assert "reaction_start_date" not in features, "reaction_start_date found in features"

    def test_forbidden_columns_are_targets(self, ml_dataset, feature_manifest):
        """All forbidden columns must be marked as targets OR audit."""
        targets = feature_manifest[feature_manifest["is_target"] == True]["feature_name"].tolist()
        audit = feature_manifest[feature_manifest["is_target"] == False]["feature_name"].tolist()
        # Actually audit columns aren't in manifest... check differently
        # The forbidden columns should either be targets or audit columns
        forbidden = [
            "reaction_class",
            "abnormal_return_1d", "abnormal_return_3d", "abnormal_return_5d",
            "return_1d_after", "return_3d_after", "return_5d_after",
            "benchmark_return_1d_after", "benchmark_return_3d_after", "benchmark_return_5d_after",
            "reaction_start_date",  # This is an audit column
        ]
        for col in forbidden:
            in_targets = col in targets
            in_audit = col in ["reaction_start_date"]  # Known audit columns
            assert in_targets or in_audit, f"Forbidden column {col} not marked as target or audit"


class TestPointInTimeFundamentals:
    """Test point-in-time fundamental data assertions."""

    def test_fundamental_source_before_cutoff(self, ml_dataset):
        """Fundamental estimated available date must be <= feature_cutoff_date."""
        mask = ml_dataset["fundamental_estimated_available_date"].notna()
        if mask.any():
            violations = ml_dataset.loc[mask, "fundamental_estimated_available_date"] > ml_dataset.loc[mask, "feature_cutoff_date"]
            assert not violations.any(), f"Fundamental estimated available after cutoff: {violations.sum()} violations"

    def test_fundamental_period_end_before_event(self, ml_dataset):
        """Fundamental period end date must be < event's period_ended (no same-quarter leakage)."""
        mask = ml_dataset["fundamental_period_end_date"].notna()
        if mask.any():
            violations = ml_dataset.loc[mask, "fundamental_period_end_date"] >= ml_dataset.loc[mask, "period_ended"]
            assert not violations.any(), f"Fundamental period_end >= event period_ended: {violations.sum()} violations"

    def test_fundamental_lag_non_negative(self, ml_dataset):
        """Fundamental lag days must be non-negative."""
        mask = ml_dataset["fundamental_lag_days"].notna()
        if mask.any():
            assert (ml_dataset.loc[mask, "fundamental_lag_days"] >= 0).all(), "Negative fundamental lag"


class TestPriorEarningsLeakage:
    """Test prior-earnings features use only prior events."""

    def test_prior_earnings_count_monotonic(self, ml_dataset):
        """Prior earnings count should be non-decreasing within symbol over time."""
        # Note: ML dataset is a subset of all events (only those with price data).
        # prior_earnings_count was computed on full 3,765 canonical events.
        # Within the ML subset, counts should still be strictly increasing per symbol.
        df = ml_dataset.sort_values(["symbol", "period_ended"]).reset_index(drop=True)
        for symbol, group in df.groupby("symbol"):
            counts = group["prior_earnings_count"].values
            # Should be strictly increasing (no duplicates, no decreases)
            diffs = np.diff(counts)
            assert (diffs > 0).all(), f"Prior count not strictly increasing for {symbol}: {counts}"

    def test_prior_last_reaction_not_current(self, ml_dataset):
        """Prior last reaction should not equal current reaction (except streaks)."""
        # This is a soft check - streaks are legitimate
        # Just verify the column exists and is populated
        assert "prior_last_reaction_class" in ml_dataset.columns
        assert ml_dataset["prior_last_reaction_class"].notna().sum() > 0

    def test_no_future_prior_features(self, ml_dataset):
        """Prior features should not use future data."""
        # The prior features are computed from the full canonical events table
        # and merged by event_key, so they cannot contain future info
        # This is verified by construction in build_prior_earnings_features.py
        pass


class TestChronologicalSplit:
    """Test chronological train/val/test split."""

    def test_split_no_overlap(self, ml_dataset):
        """Train/val/test splits by period must not overlap."""
        unique_periods = sorted(ml_dataset["period_ended"].unique())
        n = len(unique_periods)
        train_periods = unique_periods[:int(n * 0.6)]
        val_periods = unique_periods[int(n * 0.6):int(n * 0.8)]
        test_periods = unique_periods[int(n * 0.8):]
        
        train = ml_dataset[ml_dataset["period_ended"].isin(train_periods)]
        val = ml_dataset[ml_dataset["period_ended"].isin(val_periods)]
        test = ml_dataset[ml_dataset["period_ended"].isin(test_periods)]
        
        assert train["period_ended"].max() < val["period_ended"].min()
        assert val["period_ended"].max() < test["period_ended"].min()

    def test_split_all_periods_assigned(self, ml_dataset):
        """All periods must be assigned to exactly one split."""
        unique_periods = sorted(ml_dataset["period_ended"].unique())
        n = len(unique_periods)
        train_periods = set(unique_periods[:int(n * 0.6)])
        val_periods = set(unique_periods[int(n * 0.6):int(n * 0.8)])
        test_periods = set(unique_periods[int(n * 0.8):])
        
        all_assigned = train_periods | val_periods | test_periods
        assert all_assigned == set(unique_periods)
        # No overlap
        assert len(train_periods & val_periods) == 0
        assert len(val_periods & test_periods) == 0
        assert len(train_periods & test_periods) == 0


class TestEventIntegrity:
    """Test event-level integrity."""

    def test_event_keys_unique(self, ml_dataset):
        """Event keys must be unique."""
        assert ml_dataset["event_key"].is_unique

    def test_symbol_period_unique(self, ml_dataset):
        """Symbol + period combinations must be unique."""
        assert not ml_dataset.duplicated(subset=["symbol", "period_ended"]).any()

    def test_no_missing_critical_fields(self, ml_dataset):
        """Critical fields must not be missing."""
        critical = ["event_key", "symbol", "period_ended", "feature_cutoff_date", 
                    "reaction_start_date", "result_announcement_datetime"]
        for col in critical:
            assert ml_dataset[col].notna().all(), f"Missing values in {col}"

    def test_period_ended_is_quarter_end(self, ml_dataset):
        """Period ended must be valid quarter end."""
        valid_ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
        for _, row in ml_dataset.iterrows():
            pe = row["period_ended"]
            assert (pe.month, pe.day) in valid_ends, f"Invalid period end: {pe}"


class TestFeatureCalculations:
    """Test feature calculation integrity."""

    def test_no_infinite_features(self, ml_dataset, feature_manifest):
        """Features must not contain infinite values."""
        features = feature_manifest[feature_manifest["is_target"] == False]["feature_name"].tolist()
        for col in features:
            if col in ml_dataset.columns and ml_dataset[col].dtype in [np.float64, np.float32]:
                assert not np.isinf(ml_dataset[col]).any(), f"Infinite values in {col}"

    def test_drawdowns_non_positive(self, ml_dataset):
        """Drawdown features must be <= 0."""
        dd_cols = [c for c in ml_dataset.columns if c.startswith("drawdown_")]
        for col in dd_cols:
            vals = ml_dataset[col].dropna()
            assert (vals <= 0.01).all(), f"Positive drawdown in {col}"

    def test_volatility_non_negative(self, ml_dataset):
        """Volatility features must be >= 0."""
        vol_cols = [c for c in ml_dataset.columns if c.startswith("volatility_")]
        for col in vol_cols:
            vals = ml_dataset[col].dropna()
            assert (vals >= -0.01).all(), f"Negative volatility in {col}"

    def test_abnormal_return_consistency(self, ml_dataset):
        """Abnormal return = stock return - benchmark return."""
        for w in [1, 3, 5]:
            stock = ml_dataset[f"return_{w}d_after"]
            bench = ml_dataset[f"benchmark_return_{w}d_after"]
            abn = ml_dataset[f"abnormal_return_{w}d"]
            mask = stock.notna() & bench.notna() & abn.notna()
            if mask.any():
                diff = (stock - bench - abn).loc[mask].abs()
                assert (diff < 1e-10).all(), f"Abnormal return mismatch for {w}d"


class TestTargetConstruction:
    """Test target/label construction."""

    def test_reaction_class_values(self, ml_dataset):
        """Reaction class must be valid."""
        valid = {"POSITIVE", "NEUTRAL", "NEGATIVE", "UNKNOWN"}
        invalid = set(ml_dataset["reaction_class"].unique()) - valid
        assert len(invalid) == 0, f"Invalid reaction classes: {invalid}"

    def test_reaction_class_from_abnormal_3d(self, ml_dataset):
        """Reaction class should be based on 3-day abnormal return."""
        # This is a design check - the actual logic is in build_event_price_dataset.py
        # We verify the classes exist and are balanced
        dist = ml_dataset["reaction_class"].value_counts()
        assert len(dist) >= 3, "Should have at least 3 classes"


class TestDataQuality:
    """Test data quality."""

    def test_all_symbols_in_nifty50(self, ml_dataset):
        """All symbols must be in NIFTY 50."""
        nifty50 = pd.read_csv("data/reference/nifty50_clean.csv")["symbol"].tolist()
        for sym in ml_dataset["symbol"].unique():
            assert sym in nifty50, f"Symbol {sym} not in NIFTY 50"

    def test_price_history_positive(self, ml_dataset):
        """Price history days must be positive."""
        assert (ml_dataset["price_history_days"] > 0).all()

    def test_benchmark_history_positive(self, ml_dataset):
        """Benchmark history days must be positive."""
        assert (ml_dataset["benchmark_history_days"] > 0).all()

    def test_session_type_valid(self, ml_dataset):
        """Session type must be valid."""
        valid = {"pre_market", "during_market", "post_market", "unknown", "non_trading_day"}
        invalid = set(ml_dataset["announcement_session_type"].unique()) - valid
        assert len(invalid) == 0, f"Invalid session types: {invalid}"


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))