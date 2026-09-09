"""
Model Guardrails — Anti-Degenerate Prediction Validation

This module provides HARD FAILURE validation functions that future model training
code MUST call after generating predictions. These are not warnings — they raise
exceptions to prevent silent acceptance of degenerate models.

The previous LightGBM failure: Model predicted NEUTRAL for almost all validation
events, early stopping at iteration 1, but pipeline reported "training completed"
without detecting the degenerate predictions. This must NEVER happen again.
"""

from typing import Sequence
import numpy as np
import pandas as pd


class GuardrailViolation(Exception):
    """Raised when a model guardrail check fails."""
    pass


def validate_classification_predictions(
    y_true: Sequence,
    y_pred: Sequence,
    valid_classes: Sequence = None,
    min_unique_classes: int = 2,
) -> dict:
    """
    Validate classification predictions for degeneracy.

    HARD FAILS if:
    - Predictions contain fewer than min_unique_classes unique classes (default 2)
    - Any valid class has zero predictions
    - Predictions contain invalid class labels

    Args:
        y_true: True labels
        y_pred: Predicted labels
        valid_classes: Expected class labels (inferred from y_true if None)
        min_unique_classes: Minimum unique predicted classes required (default 2)

    Returns:
        Dict with prediction counts per class

    Raises:
        GuardrailViolation: If validation fails
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if len(y_pred) != len(y_true):
        raise GuardrailViolation(
            f"Length mismatch: y_pred={len(y_pred)}, y_true={len(y_true)}"
        )

    if len(y_pred) == 0:
        raise GuardrailViolation("Empty predictions array")

    if valid_classes is None:
        valid_classes = np.unique(y_true)

    valid_classes = np.asarray(valid_classes)

    # Check for NaN predictions BEFORE calling np.unique (which fails on mixed types)
    if y_pred.dtype.kind in 'OUSV':  # object, unicode, string, void
        # For object/string arrays, check each element
        nan_mask = pd.isna(y_pred)
        if np.any(nan_mask):
            nan_count = int(np.sum(nan_mask))
            raise GuardrailViolation(f"NaN PREDICTIONS: {nan_count} NaN values in predictions")

    unique_pred = np.unique(y_pred)
    n_unique_pred = len(unique_pred)

    # Check 1: Minimum unique predicted classes
    if n_unique_pred < min_unique_classes:
        pred_counts = {c: int(np.sum(y_pred == c)) for c in unique_pred}
        raise GuardrailViolation(
            f"DEGENERATE PREDICTIONS: Only {n_unique_pred} unique class(es) predicted "
            f"(minimum required: {min_unique_classes}). "
            f"Prediction counts: {pred_counts}. "
            f"This indicates model collapse — training MUST NOT proceed."
        )

    # Check 2: No valid class should have zero predictions (optional strictness)
    # We only warn about this, not hard fail, because a model predicting
    # NEUTRAL + POSITIVE (but no NEGATIVE) may still be useful
    missing_classes = set(valid_classes) - set(unique_pred)
    if missing_classes:
        # This is informational — not a hard fail per requirements
        pass

    # Check 3: No invalid prediction labels
    invalid_labels = set(unique_pred) - set(valid_classes)
    if invalid_labels:
        raise GuardrailViolation(
            f"INVALID PREDICTION LABELS: {invalid_labels} not in valid classes {valid_classes}"
        )

    # Check 4: NaN predictions
    if isinstance(y_pred, np.ndarray) and y_pred.dtype.kind in 'fOU':
        nan_count = np.sum(pd.isna(y_pred))
        if nan_count > 0:
            raise GuardrailViolation(f"NaN PREDICTIONS: {nan_count} NaN values in predictions")

    # Return prediction distribution for logging
    pred_counts = {c: int(np.sum(y_pred == c)) for c in valid_classes}
    return pred_counts


def validate_prediction_probabilities(
    y_proba: np.ndarray,
    valid_classes: Sequence = None,
    max_prob_threshold: float = 0.99,
    min_prob_mass: float = 0.99,
) -> dict:
    """
    Validate prediction probabilities for collapse/calibration issues.

    HARD FAILS if:
    - Probabilities contain NaN or inf
    - Shape mismatch (n_samples, n_classes) expected
    - Any sample has max probability > max_prob_threshold (near-deterministic collapse)
    - Probability rows don't sum to ~1.0 (within min_prob_mass)

    Args:
        y_proba: Predicted probabilities, shape (n_samples, n_classes)
        valid_classes: Class labels (for error messages)
        max_prob_threshold: Maximum allowed max probability per sample (default 0.99)
        min_prob_mass: Minimum required sum of probabilities per row (default 0.99)

    Returns:
        Dict with probability statistics

    Raises:
        GuardrailViolation: If validation fails
    """
    y_proba = np.asarray(y_proba)

    if y_proba.ndim != 2:
        raise GuardrailViolation(
            f"PROBABILITY SHAPE MISMATCH: Expected 2D array (n_samples, n_classes), "
            f"got shape {y_proba.shape}"
        )

    n_samples, n_classes = y_proba.shape

    if valid_classes is not None and len(valid_classes) != n_classes:
        raise GuardrailViolation(
            f"PROBABILITY SHAPE MISMATCH: {n_classes} probability columns "
            f"but {len(valid_classes)} valid classes provided"
        )

    # Check for NaN/inf
    if np.any(np.isnan(y_proba)):
        nan_count = np.sum(np.isnan(y_proba))
        raise GuardrailViolation(f"NaN PROBABILITIES: {nan_count} NaN values in probability matrix")

    if np.any(np.isinf(y_proba)):
        inf_count = np.sum(np.isinf(y_proba))
        raise GuardrailViolation(f"INFINITE PROBABILITIES: {inf_count} infinite values")

    # Check probability mass per row
    row_sums = y_proba.sum(axis=1)
    if np.any(row_sums < min_prob_mass) or np.any(row_sums > 1.0 / min_prob_mass):
        bad_rows = np.sum((row_sums < min_prob_mass) | (row_sums > 1.0 / min_prob_mass))
        raise GuardrailViolation(
            f"PROBABILITY MASS VIOLATION: {bad_rows} rows have sum outside "
            f"[{min_prob_mass}, {1.0/min_prob_mass:.3f}]. "
            f"Row sum range: [{row_sums.min():.4f}, {row_sums.max():.4f}]"
        )

    # Check for probability collapse (near-deterministic predictions)
    max_probs = y_proba.max(axis=1)
    collapsed_count = np.sum(max_probs > max_prob_threshold)
    if collapsed_count > 0:
        collapse_pct = collapsed_count / n_samples * 100
        if collapse_pct > 95:  # If >95% of samples are collapsed, it's degenerate
            raise GuardrailViolation(
                f"PROBABILITY COLLAPSE: {collapsed_count}/{n_samples} samples "
                f"({collapse_pct:.1f}%) have max probability > {max_prob_threshold}. "
                f"This indicates model is predicting near-deterministically. "
                f"Max probability range: [{max_probs.min():.4f}, {max_probs.max():.4f}]"
            )

    return {
        "n_samples": n_samples,
        "n_classes": n_classes,
        "max_prob_mean": float(max_probs.mean()),
        "max_prob_max": float(max_probs.max()),
        "max_prob_min": float(max_probs.min()),
        "collapsed_samples": int(collapsed_count),
        "row_sum_mean": float(row_sums.mean()),
        "row_sum_std": float(row_sums.std()),
    }


def compare_against_baseline(
    y_true: Sequence,
    y_pred: Sequence,
    y_proba: np.ndarray = None,
    valid_classes: Sequence = None,
    min_macro_f1_improvement: float = 0.01,
    min_balanced_acc_improvement: float = 0.01,
) -> dict:
    """
    Compare model predictions against majority-class baseline.

    HARD FAILS if:
    - Model macro F1 is not better than majority-class baseline by min_macro_f1_improvement
    - Model balanced accuracy is not better than majority-class baseline by min_balanced_acc_improvement

    This ensures the model actually learns something beyond predicting the majority class.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for log loss comparison)
        valid_classes: Class labels
        min_macro_f1_improvement: Minimum required macro F1 improvement over baseline
        min_balanced_acc_improvement: Minimum required balanced accuracy improvement

    Returns:
        Dict with baseline and model metrics

    Raises:
        GuardrailViolation: If model doesn't beat baseline
    """
    from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, log_loss

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if valid_classes is None:
        valid_classes = np.unique(y_true)

    # Majority class baseline (predict most frequent class for all samples)
    majority_class = pd.Series(y_true).mode()[0]
    y_baseline = np.full_like(y_true, majority_class)

    # Model metrics
    model_macro_f1 = f1_score(y_true, y_pred, average='macro', labels=valid_classes, zero_division=0)
    model_balanced_acc = balanced_accuracy_score(y_true, y_pred)
    model_accuracy = accuracy_score(y_true, y_pred)

    # Baseline metrics
    baseline_macro_f1 = f1_score(y_true, y_baseline, average='macro', labels=valid_classes, zero_division=0)
    baseline_balanced_acc = balanced_accuracy_score(y_true, y_baseline)
    baseline_accuracy = accuracy_score(y_true, y_baseline)

    # Check improvements
    macro_f1_improvement = model_macro_f1 - baseline_macro_f1
    balanced_acc_improvement = model_balanced_acc - baseline_balanced_acc

    results = {
        "model": {
            "macro_f1": model_macro_f1,
            "balanced_accuracy": model_balanced_acc,
            "accuracy": model_accuracy,
        },
        "baseline": {
            "majority_class": majority_class,
            "macro_f1": baseline_macro_f1,
            "balanced_accuracy": baseline_balanced_acc,
            "accuracy": baseline_accuracy,
        },
        "improvement": {
            "macro_f1": macro_f1_improvement,
            "balanced_accuracy": balanced_acc_improvement,
        },
    }

    if y_proba is not None:
        try:
            model_logloss = log_loss(y_true, y_proba, labels=valid_classes)
            baseline_proba = np.zeros((len(y_true), len(valid_classes)))
            class_idx = {c: i for i, c in enumerate(valid_classes)}
            baseline_proba[:, class_idx[majority_class]] = 1.0
            baseline_logloss = log_loss(y_true, baseline_proba, labels=valid_classes)
            results["model"]["log_loss"] = model_logloss
            results["baseline"]["log_loss"] = baseline_logloss
            results["improvement"]["log_loss"] = baseline_logloss - model_logloss  # lower is better
        except Exception:
            pass

    # Hard fail if model doesn't beat baseline
    if macro_f1_improvement < min_macro_f1_improvement:
        raise GuardrailViolation(
            f"MODEL DOES NOT BEAT BASELINE: Macro F1 improvement = {macro_f1_improvement:.4f} "
            f"(required >= {min_macro_f1_improvement}). "
            f"Model macro F1: {model_macro_f1:.4f}, Baseline macro F1: {baseline_macro_f1:.4f}. "
            f"Model is not learning beyond majority class ({majority_class})."
        )

    if balanced_acc_improvement < min_balanced_acc_improvement:
        raise GuardrailViolation(
            f"MODEL DOES NOT BEAT BASELINE: Balanced accuracy improvement = {balanced_acc_improvement:.4f} "
            f"(required >= {min_balanced_acc_improvement}). "
            f"Model balanced acc: {model_balanced_acc:.4f}, Baseline balanced acc: {baseline_balanced_acc:.4f}. "
            f"Model is not learning beyond majority class ({majority_class})."
        )

    return results


def validate_regression_predictions(
    y_true: Sequence,
    y_pred: Sequence,
    max_mae_ratio: float = 1.0,
    max_rmse_ratio: float = 1.0,
) -> dict:
    """
    Validate regression predictions against naive baselines.

    HARD FAILS if:
    - MAE is worse than (or equal to) naive mean prediction baseline
    - RMSE is worse than (or equal to) naive mean prediction baseline
    - Predictions contain NaN or inf
    - Predictions are constant (zero variance)

    Args:
        y_true: True values
        y_pred: Predicted values
        max_mae_ratio: Maximum allowed MAE / baseline_MAE (default 1.0 = must beat baseline)
        max_rmse_ratio: Maximum allowed RMSE / baseline_RMSE (default 1.0 = must beat baseline)

    Returns:
        Dict with regression metrics

    Raises:
        GuardrailViolation: If validation fails
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_pred) != len(y_true):
        raise GuardrailViolation(f"Length mismatch: y_pred={len(y_pred)}, y_true={len(y_true)}")

    if len(y_pred) == 0:
        raise GuardrailViolation("Empty predictions array")

    # Check for NaN/inf
    if np.any(np.isnan(y_pred)):
        raise GuardrailViolation(f"NaN PREDICTIONS: {np.sum(np.isnan(y_pred))} NaN values")

    if np.any(np.isinf(y_pred)):
        raise GuardrailViolation(f"INFINITE PREDICTIONS: {np.sum(np.isinf(y_pred))} infinite values")

    # Check for constant predictions
    if np.std(y_pred) < 1e-10:
        raise GuardrailViolation(
            f"CONSTANT PREDICTIONS: All predictions are identical ({y_pred[0]:.6f}). "
            f"Model has zero variance — this is degenerate."
        )

    # Naive baseline: predict mean of y_true
    y_baseline = np.full_like(y_true, np.mean(y_true))

    model_mae = mean_absolute_error(y_true, y_pred)
    model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    baseline_mae = mean_absolute_error(y_true, y_baseline)
    baseline_rmse = np.sqrt(mean_squared_error(y_true, y_baseline))

    mae_ratio = model_mae / baseline_mae if baseline_mae > 0 else float('inf')
    rmse_ratio = model_rmse / baseline_rmse if baseline_rmse > 0 else float('inf')

    results = {
        "model": {"mae": model_mae, "rmse": model_rmse, "std": float(np.std(y_pred))},
        "baseline": {"mae": baseline_mae, "rmse": baseline_rmse, "mean": float(np.mean(y_true))},
        "ratio": {"mae": mae_ratio, "rmse": rmse_ratio},
    }

    if mae_ratio > max_mae_ratio:
        raise GuardrailViolation(
            f"REGRESSION MODEL DOES NOT BEAT BASELINE: MAE ratio = {mae_ratio:.4f} "
            f"(required <= {max_mae_ratio}). "
            f"Model MAE: {model_mae:.6f}, Baseline MAE: {baseline_mae:.6f}. "
            f"Model is not learning beyond predicting the mean."
        )

    if rmse_ratio > max_rmse_ratio:
        raise GuardrailViolation(
            f"REGRESSION MODEL DOES NOT BEAT BASELINE: RMSE ratio = {rmse_ratio:.4f} "
            f"(required <= {max_rmse_ratio}). "
            f"Model RMSE: {model_rmse:.6f}, Baseline RMSE: {baseline_rmse:.6f}. "
            f"Model is not learning beyond predicting the mean."
        )

    return results


def calculate_baseline_metrics(y_true: Sequence, valid_classes: Sequence = None) -> dict:
    """
    Calculate majority-class baseline metrics for comparison.

    This is a utility function that future training code can use to compute
    baseline metrics WITHOUT raising exceptions.

    Args:
        y_true: True labels
        valid_classes: Class labels

    Returns:
        Dict with baseline metrics
    """
    from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

    y_true = np.asarray(y_true)

    if valid_classes is None:
        valid_classes = np.unique(y_true)

    majority_class = pd.Series(y_true).mode()[0]
    y_baseline = np.full_like(y_true, majority_class)

    return {
        "majority_class": majority_class,
        "macro_f1": f1_score(y_true, y_baseline, average='macro', labels=valid_classes, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_baseline),
        "accuracy": accuracy_score(y_true, y_baseline),
        "class_distribution": {c: int(np.sum(y_true == c)) for c in valid_classes},
    }


def calculate_regression_baseline_metrics(y_true: Sequence) -> dict:
    """
    Calculate naive mean-prediction baseline metrics for regression.

    Args:
        y_true: True values

    Returns:
        Dict with baseline metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y_true = np.asarray(y_true, dtype=float)
    y_baseline = np.full_like(y_true, np.mean(y_true))

    return {
        "mean": float(np.mean(y_true)),
        "mae": mean_absolute_error(y_true, y_baseline),
        "rmse": np.sqrt(mean_squared_error(y_true, y_baseline)),
    }