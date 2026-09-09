"""
Regression Tests for Model Guardrails

These tests verify that the guardrail functions correctly catch the previous
LightGBM failure mode (predicting only NEUTRAL) and other degenerate cases.

This test MUST PASS to prove that a future model-training pipeline cannot
silently accept the previous failure.
"""

import numpy as np
import pandas as pd
import pytest
from scripts.model_guardrails import (
    GuardrailViolation,
    validate_classification_predictions,
    validate_prediction_probabilities,
    compare_against_baseline,
    validate_regression_predictions,
    calculate_baseline_metrics,
    calculate_regression_baseline_metrics,
)


class TestValidateClassificationPredictions:
    """Tests for classification prediction validation."""

    def setup_method(self):
        self.valid_classes = np.array(['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
        self.y_true = np.array(['NEUTRAL'] * 200 + ['POSITIVE'] * 100 + ['NEGATIVE'] * 100)

    def test_all_neutral_predictions_raises(self):
        """Test that all-NEUTRAL predictions hard fails (the previous LightGBM failure)."""
        y_pred = np.array(['NEUTRAL'] * 400)

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_classification_predictions(self.y_true, y_pred, self.valid_classes)

        assert "DEGENERATE PREDICTIONS" in str(exc_info.value)
        assert "Only 1 unique class" in str(exc_info.value)

    def test_all_positive_predictions_raises(self):
        """Test that all-POSITIVE predictions hard fails."""
        y_pred = np.array(['POSITIVE'] * 400)

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_classification_predictions(self.y_true, y_pred, self.valid_classes)

        assert "DEGENERATE PREDICTIONS" in str(exc_info.value)

    def test_all_negative_predictions_raises(self):
        """Test that all-NEGATIVE predictions hard fails."""
        y_pred = np.array(['NEGATIVE'] * 400)

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_classification_predictions(self.y_true, y_pred, self.valid_classes)

        assert "DEGENERATE PREDICTIONS" in str(exc_info.value)

    def test_neutral_plus_positive_passes(self):
        """Test that NEUTRAL + POSITIVE predictions passes (2 classes minimum)."""
        y_pred = np.array(['NEUTRAL'] * 250 + ['POSITIVE'] * 150)

        result = validate_classification_predictions(self.y_true, y_pred, self.valid_classes)
        assert result['NEUTRAL'] == 250
        assert result['POSITIVE'] == 150
        assert result['NEGATIVE'] == 0

    def test_neutral_plus_negative_passes(self):
        """Test that NEUTRAL + NEGATIVE predictions passes."""
        y_pred = np.array(['NEUTRAL'] * 250 + ['NEGATIVE'] * 150)

        result = validate_classification_predictions(self.y_true, y_pred, self.valid_classes)
        assert result['NEUTRAL'] == 250
        assert result['NEGATIVE'] == 150

    def test_all_three_classes_passes(self):
        """Test that all three classes predicted passes."""
        y_pred = np.array(['NEUTRAL'] * 150 + ['POSITIVE'] * 150 + ['NEGATIVE'] * 100)

        result = validate_classification_predictions(self.y_true, y_pred, self.valid_classes)
        assert result['NEUTRAL'] == 150
        assert result['POSITIVE'] == 150
        assert result['NEGATIVE'] == 100

    def test_invalid_prediction_label_raises(self):
        """Test that invalid prediction labels raise."""
        y_pred = np.array(['NEUTRAL'] * 200 + ['INVALID_CLASS'] * 200)

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_classification_predictions(self.y_true, y_pred, self.valid_classes)

        assert "INVALID PREDICTION LABELS" in str(exc_info.value)

    def test_nan_predictions_raises(self):
        """Test that NaN predictions raise."""
        # Use pandas Series with NaN to preserve NaN properly
        y_pred = pd.Series(['NEUTRAL'] * 200 + ['POSITIVE'] * 199 + [np.nan])

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_classification_predictions(self.y_true, y_pred, self.valid_classes)

        assert "NaN PREDICTIONS" in str(exc_info.value)

    def test_empty_predictions_raises(self):
        """Test that empty predictions raise."""
        with pytest.raises(GuardrailViolation) as exc_info:
            validate_classification_predictions([], [], self.valid_classes)

        assert "Empty predictions" in str(exc_info.value)

    def test_length_mismatch_raises(self):
        """Test that length mismatch raises."""
        with pytest.raises(GuardrailViolation) as exc_info:
            validate_classification_predictions(
                self.y_true[:100], np.array(['NEUTRAL'] * 50), self.valid_classes
            )

        assert "Length mismatch" in str(exc_info.value)


class TestValidatePredictionProbabilities:
    """Tests for prediction probability validation."""

    def setup_method(self):
        self.valid_classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

    def test_nan_probabilities_raises(self):
        """Test that NaN probabilities raise."""
        y_proba = np.array([
            [0.3, 0.4, 0.3],
            [np.nan, 0.5, 0.5],
            [0.2, 0.3, 0.5],
        ])

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_prediction_probabilities(y_proba, self.valid_classes)

        assert "NaN PROBABILITIES" in str(exc_info.value)

    def test_infinite_probabilities_raises(self):
        """Test that infinite probabilities raise."""
        y_proba = np.array([
            [0.3, 0.4, 0.3],
            [np.inf, 0.5, 0.5],
            [0.2, 0.3, 0.5],
        ])

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_prediction_probabilities(y_proba, self.valid_classes)

        assert "INFINITE PROBABILITIES" in str(exc_info.value)

    def test_shape_mismatch_raises(self):
        """Test that wrong shape raises."""
        y_proba = np.array([0.3, 0.4, 0.3])  # 1D instead of 2D

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_prediction_probabilities(y_proba, self.valid_classes)

        assert "PROBABILITY SHAPE MISMATCH" in str(exc_info.value)

    def test_probability_columns_mismatch_raises(self):
        """Test that class count mismatch raises."""
        y_proba = np.array([
            [0.3, 0.4],  # Only 2 columns
            [0.2, 0.8],
        ])
        valid_classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']  # 3 classes

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_prediction_probabilities(y_proba, valid_classes)

        assert "PROBABILITY SHAPE MISMATCH" in str(exc_info.value)

    def test_probability_mass_violation_raises(self):
        """Test that probability rows not summing to ~1 raise."""
        y_proba = np.array([
            [0.3, 0.4, 0.3],  # sums to 1.0
            [0.2, 0.3, 0.3],  # sums to 0.8 - violation
        ])

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_prediction_probabilities(y_proba, self.valid_classes)

        assert "PROBABILITY MASS VIOLATION" in str(exc_info.value)

    def test_probability_collapse_raises(self):
        """Test that near-deterministic probabilities raise (collapse)."""
        # 100 samples, all with max probability > 0.99
        y_proba = np.zeros((100, 3))
        y_proba[:, 1] = 0.995  # All predict NEUTRAL with 99.5% confidence
        y_proba[:, 0] = 0.0025
        y_proba[:, 2] = 0.0025

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_prediction_probabilities(y_proba, self.valid_classes)

        assert "PROBABILITY COLLAPSE" in str(exc_info.value)

    def test_valid_probabilities_passes(self):
        """Test that well-calibrated probabilities pass."""
        y_proba = np.array([
            [0.2, 0.5, 0.3],
            [0.3, 0.4, 0.3],
            [0.1, 0.2, 0.7],
            [0.4, 0.4, 0.2],
        ])

        result = validate_prediction_probabilities(y_proba, self.valid_classes)
        assert result['n_samples'] == 4
        assert result['n_classes'] == 3


class TestCompareAgainstBaseline:
    """Tests for baseline comparison."""

    def setup_method(self):
        self.valid_classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        # Imbalanced: 60% NEUTRAL, 20% POSITIVE, 20% NEGATIVE
        self.y_true = np.array(
            ['NEUTRAL'] * 600 + ['POSITIVE'] * 200 + ['NEGATIVE'] * 200
        )

    def test_model_same_as_baseline_raises(self):
        """Test that model matching majority-class baseline fails."""
        # Model predicts NEUTRAL for everything (same as baseline)
        y_pred = np.array(['NEUTRAL'] * 1000)

        with pytest.raises(GuardrailViolation) as exc_info:
            compare_against_baseline(self.y_true, y_pred, valid_classes=self.valid_classes)

        assert "MODEL DOES NOT BEAT BASELINE" in str(exc_info.value)
        assert "Macro F1 improvement" in str(exc_info.value)

    def test_model_worse_than_baseline_raises(self):
        """Test that model worse than baseline fails."""
        # Model predicts POSITIVE for everything (worse than NEUTRAL baseline)
        y_pred = np.array(['POSITIVE'] * 1000)

        with pytest.raises(GuardrailViolation) as exc_info:
            compare_against_baseline(self.y_true, y_pred, valid_classes=self.valid_classes)

        assert "MODEL DOES NOT BEAT BASELINE" in str(exc_info.value)

    def test_model_beats_baseline_passes(self):
        """Test that model beating baseline passes."""
        # Model gets some right: 500 NEUTRAL, 150 POSITIVE, 150 NEGATIVE (rest wrong)
        y_pred = np.array(
            ['NEUTRAL'] * 500 + ['POSITIVE'] * 150 + ['NEGATIVE'] * 150 + ['NEUTRAL'] * 200
        )

        result = compare_against_baseline(
            self.y_true, y_pred, valid_classes=self.valid_classes,
            min_macro_f1_improvement=0.01,
            min_balanced_acc_improvement=0.01
        )
        assert result['improvement']['macro_f1'] > 0.01
        assert result['improvement']['balanced_accuracy'] > 0.01

    def test_perfect_predictions_passes(self):
        """Test that perfect predictions pass."""
        y_pred = self.y_true.copy()

        result = compare_against_baseline(self.y_true, y_pred, valid_classes=self.valid_classes)
        assert result['model']['macro_f1'] == 1.0
        assert result['model']['balanced_accuracy'] == 1.0


class TestValidateRegressionPredictions:
    """Tests for regression prediction validation."""

    def setup_method(self):
        np.random.seed(42)
        self.y_true = np.random.normal(0, 1, 1000)

    def test_constant_predictions_raises(self):
        """Test that constant predictions raise."""
        y_pred = np.full(1000, 0.5)

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_regression_predictions(self.y_true, y_pred)

        assert "CONSTANT PREDICTIONS" in str(exc_info.value)

    def test_nan_predictions_raises(self):
        """Test that NaN predictions raise."""
        y_pred = np.random.normal(0, 1, 1000)
        y_pred[0] = np.nan

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_regression_predictions(self.y_true, y_pred)

        assert "NaN PREDICTIONS" in str(exc_info.value)

    def test_infinite_predictions_raises(self):
        """Test that infinite predictions raise."""
        y_pred = np.random.normal(0, 1, 1000)
        y_pred[0] = np.inf

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_regression_predictions(self.y_true, y_pred)

        assert "INFINITE PREDICTIONS" in str(exc_info.value)

    def test_worse_than_mean_baseline_raises(self):
        """Test that predictions worse than mean baseline raise."""
        # Predict values with tiny variance but far from mean
        # This avoids the constant prediction check but is still worse than baseline
        y_pred = np.full(1000, 10.0) + np.random.normal(0, 0.001, 1000)

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_regression_predictions(self.y_true, y_pred)

        assert "REGRESSION MODEL DOES NOT BEAT BASELINE" in str(exc_info.value)

    def test_better_than_baseline_passes(self):
        """Test that predictions better than baseline pass."""
        # Add small noise to true values
        y_pred = self.y_true + np.random.normal(0, 0.1, 1000)

        result = validate_regression_predictions(self.y_true, y_pred)
        assert result['ratio']['mae'] < 1.0
        assert result['ratio']['rmse'] < 1.0


class TestCalculateBaselineMetrics:
    """Tests for baseline metric calculation utilities."""

    def test_classification_baseline(self):
        """Test classification baseline calculation."""
        y_true = np.array(['NEUTRAL'] * 600 + ['POSITIVE'] * 200 + ['NEGATIVE'] * 200)
        valid_classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

        result = calculate_baseline_metrics(y_true, valid_classes)

        assert result['majority_class'] == 'NEUTRAL'
        assert result['accuracy'] == 0.6
        assert result['class_distribution']['NEUTRAL'] == 600

    def test_regression_baseline(self):
        """Test regression baseline calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = calculate_regression_baseline_metrics(y_true)

        assert result['mean'] == 3.0
        # MAE of mean prediction: |1-3|+|2-3|+|3-3|+|4-3|+|5-3| / 5 = 6/5 = 1.2
        assert abs(result['mae'] - 1.2) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])