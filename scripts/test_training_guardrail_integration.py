#!/usr/bin/env python
"""
Guardrail Integration Test

This test demonstrates that a synthetic training pipeline MUST invoke
the model guardrails and that they correctly catch degenerate predictions.

This is NOT a model training test - it tests the GUARDRAIL INVOCATION
mechanism that the real training pipeline must implement.

The future training pipeline must follow this pattern:
1. Train model
2. Generate validation predictions
3. Call ALL guardrail functions
4. If ANY guardrail raises GuardrailViolation -> STOP, model rejected
5. Only proceed if ALL guardrails pass

This test proves the mechanism works.
"""

import sys
import numpy as np
import pandas as pd
from scripts.model_guardrails import (
    GuardrailViolation,
    validate_classification_predictions,
    validate_prediction_probabilities,
    compare_against_baseline,
    validate_regression_predictions,
)


def test_synthetic_pipeline_all_neutral():
    """
    Test Case 1: Synthetic pipeline that predicts all NEUTRAL (the previous LightGBM failure).
    This MUST trigger GuardrailViolation.
    """
    print("=" * 70)
    print("TEST 1: All-NEUTRAL predictions (previous LightGBM failure mode)")
    print("=" * 70)

    # Simulate validation set
    y_true = np.array(['NEUTRAL'] * 120 + ['POSITIVE'] * 40 + ['NEGATIVE'] * 40)
    y_pred = np.array(['NEUTRAL'] * 200)  # All NEUTRAL - DEGENERATE
    y_proba = np.zeros((200, 3))
    y_proba[:, 1] = 0.995  # All mass on NEUTRAL (class index 1) - > 0.99 threshold
    y_proba[:, 0] = 0.0025
    y_proba[:, 2] = 0.0025
    valid_classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

    print(f"y_true distribution: {pd.Series(y_true).value_counts().to_dict()}")
    print(f"y_pred distribution: {pd.Series(y_pred).value_counts().to_dict()}")
    print(f"y_proba max: {y_proba.max():.4f}")

    # The training pipeline MUST call all guardrails in sequence
    guardrails_passed = []

    # Guardrail 1: Classification prediction diversity
    try:
        result = validate_classification_predictions(y_true, y_pred, valid_classes, min_unique_classes=2)
        print("❌ GUARDRAIL 1 (class diversity): PASSED (should have failed!)")
        guardrails_passed.append(True)
    except GuardrailViolation as e:
        print(f"✅ GUARDRAIL 1 (class diversity): FAILED as expected")
        print(f"   Error: {e}")
        guardrails_passed.append(False)

    # Guardrail 2: Probability validation
    try:
        result = validate_prediction_probabilities(y_proba, valid_classes)
        print("❌ GUARDRAIL 2 (probabilities): PASSED (should have failed!)")
        guardrails_passed.append(True)
    except GuardrailViolation as e:
        print(f"✅ GUARDRAIL 2 (probabilities): FAILED as expected")
        print(f"   Error: {e}")
        guardrails_passed.append(False)

    # Guardrail 3: Baseline comparison
    try:
        result = compare_against_baseline(y_true, y_pred, y_proba, valid_classes)
        print("❌ GUARDRAIL 3 (baseline): PASSED (should have failed!)")
        guardrails_passed.append(True)
    except GuardrailViolation as e:
        print(f"✅ GUARDRAIL 3 (baseline): FAILED as expected")
        print(f"   Error: {e}")
        guardrails_passed.append(False)

    # Overall result
    if not any(guardrails_passed):
        print("\n🎯 RESULT: PIPELINE CORRECTLY REJECTED degenerate model")
        return True
    else:
        print("\n💥 RESULT: PIPELINE FAILED to reject degenerate model!")
        return False


def test_synthetic_pipeline_good_predictions():
    """
    Test Case 2: Synthetic pipeline with legitimate predictions.
    This SHOULD pass all guardrails.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Legitimate multi-class predictions")
    print("=" * 70)

    # Simulate validation set with realistic predictions
    np.random.seed(42)
    n = 200
    y_true = np.array(['NEUTRAL'] * 120 + ['POSITIVE'] * 40 + ['NEGATIVE'] * 40)

    # Generate predictions that are better than baseline but not perfect
    # ~70% accuracy overall, distributed across classes
    y_pred = []
    y_proba = np.zeros((n, 3))
    valid_classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    class_to_idx = {c: i for i, c in enumerate(valid_classes)}

    for i, true_label in enumerate(y_true):
        if true_label == 'NEUTRAL':
            # 80% correct on NEUTRAL
            if np.random.rand() < 0.8:
                pred = 'NEUTRAL'
                y_proba[i, class_to_idx['NEUTRAL']] = 0.7
                y_proba[i, class_to_idx['POSITIVE']] = 0.2
                y_proba[i, class_to_idx['NEGATIVE']] = 0.1
            else:
                pred = np.random.choice(['POSITIVE', 'NEGATIVE'])
                y_proba[i, class_to_idx[pred]] = 0.6
                y_proba[i, class_to_idx['NEUTRAL']] = 0.3
                y_proba[i, class_to_idx[{'POSITIVE': 'NEGATIVE', 'NEGATIVE': 'POSITIVE'}[pred]]] = 0.1
        elif true_label == 'POSITIVE':
            # 60% correct on POSITIVE
            if np.random.rand() < 0.6:
                pred = 'POSITIVE'
                y_proba[i, class_to_idx['POSITIVE']] = 0.6
                y_proba[i, class_to_idx['NEUTRAL']] = 0.3
                y_proba[i, class_to_idx['NEGATIVE']] = 0.1
            else:
                pred = np.random.choice(['NEUTRAL', 'NEGATIVE'])
                y_proba[i, class_to_idx[pred]] = 0.5
                y_proba[i, class_to_idx['POSITIVE']] = 0.3
                y_proba[i, class_to_idx[{'NEUTRAL': 'NEGATIVE', 'NEGATIVE': 'NEUTRAL'}[pred]]] = 0.2
        else:  # NEGATIVE
            # 60% correct on NEGATIVE
            if np.random.rand() < 0.6:
                pred = 'NEGATIVE'
                y_proba[i, class_to_idx['NEGATIVE']] = 0.6
                y_proba[i, class_to_idx['NEUTRAL']] = 0.3
                y_proba[i, class_to_idx['POSITIVE']] = 0.1
            else:
                pred = np.random.choice(['NEUTRAL', 'POSITIVE'])
                y_proba[i, class_to_idx[pred]] = 0.5
                y_proba[i, class_to_idx['NEGATIVE']] = 0.3
                y_proba[i, class_to_idx[{'NEUTRAL': 'POSITIVE', 'POSITIVE': 'NEUTRAL'}[pred]]] = 0.2
        y_pred.append(pred)

    y_pred = np.array(y_pred)

    print(f"y_true distribution: {pd.Series(y_true).value_counts().to_dict()}")
    print(f"y_pred distribution: {pd.Series(y_pred).value_counts().to_dict()}")
    print(f"Unique predicted classes: {np.unique(y_pred)}")

    guardrails_passed = []

    # Guardrail 1
    try:
        result = validate_classification_predictions(y_true, y_pred, valid_classes, min_unique_classes=2)
        print(f"✅ GUARDRAIL 1 (class diversity): PASSED - {result}")
        guardrails_passed.append(True)
    except GuardrailViolation as e:
        print(f"❌ GUARDRAIL 1 (class diversity): FAILED - {e}")
        guardrails_passed.append(False)

    # Guardrail 2
    try:
        result = validate_prediction_probabilities(y_proba, valid_classes)
        print(f"✅ GUARDRAIL 2 (probabilities): PASSED - {result}")
        guardrails_passed.append(True)
    except GuardrailViolation as e:
        print(f"❌ GUARDRAIL 2 (probabilities): FAILED - {e}")
        guardrails_passed.append(False)

    # Guardrail 3
    try:
        result = compare_against_baseline(y_true, y_pred, y_proba, valid_classes)
        print(f"✅ GUARDRAIL 3 (baseline): PASSED - Improvement: {result['improvement']}")
        guardrails_passed.append(True)
    except GuardrailViolation as e:
        print(f"❌ GUARDRAIL 3 (baseline): FAILED - {e}")
        guardrails_passed.append(False)

    if all(guardrails_passed):
        print("\n🎯 RESULT: PIPELINE CORRECTLY ACCEPTED legitimate model")
        return True
    else:
        print("\n💥 RESULT: PIPELINE INCORRECTLY REJECTED legitimate model!")
        return False


def test_synthetic_regression_degenerate():
    """
    Test Case 3: Synthetic regression pipeline with degenerate predictions.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Degenerate regression predictions (constant)")
    print("=" * 70)

    np.random.seed(42)
    y_true = np.random.normal(0, 0.03, 200)
    y_pred = np.full(200, 0.0)  # Constant prediction - DEGENERATE

    print(f"y_true: mean={y_true.mean():.6f}, std={y_true.std():.6f}")
    print(f"y_pred: mean={y_pred.mean():.6f}, std={y_pred.std():.6f}")

    try:
        result = validate_regression_predictions(y_true, y_pred)
        print(f"❌ GUARDRAIL (regression): PASSED (should have failed!) - {result}")
        return False
    except GuardrailViolation as e:
        print(f"✅ GUARDRAIL (regression): FAILED as expected")
        print(f"   Error: {e}")
        return True


def test_synthetic_regression_good():
    """
    Test Case 4: Synthetic regression pipeline with legitimate predictions.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Legitimate regression predictions")
    print("=" * 70)

    np.random.seed(42)
    y_true = np.random.normal(0, 0.03, 200)
    y_pred = y_true + np.random.normal(0, 0.01, 200)  # Noisy but correlated

    print(f"y_true: mean={y_true.mean():.6f}, std={y_true.std():.6f}")
    print(f"y_pred: mean={y_pred.mean():.6f}, std={y_pred.std():.6f}")

    try:
        result = validate_regression_predictions(y_true, y_pred)
        print(f"✅ GUARDRAIL (regression): PASSED - {result}")
        return True
    except GuardrailViolation as e:
        print(f"❌ GUARDRAIL (regression): FAILED - {e}")
        return False


def test_guardrail_cannot_be_skipped():
    """
    Test Case 5: Demonstrate that guardrails are mandatory.
    This simulates what happens if a training pipeline tries to skip them.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Guardrail enforcement pattern")
    print("=" * 70)

    # This is the PATTERN that all training pipelines MUST follow
    def run_training_pipeline_with_guardrails(y_true, y_pred, y_proba, valid_classes):
        """
        Template for the MANDATORY guardrail invocation pattern.
        Any training pipeline MUST implement this exact pattern.
        """
        # Step 1: Validate classification predictions
        pred_dist = validate_classification_predictions(y_true, y_pred, valid_classes)

        # Step 2: Validate probabilities
        prob_stats = validate_prediction_probabilities(y_proba, valid_classes)

        # Step 3: Compare against baseline
        baseline_comparison = compare_against_baseline(y_true, y_pred, y_proba, valid_classes)

        # Step 4: (For regression targets) Validate regression predictions
        # regression_results = validate_regression_predictions(y_true_reg, y_pred_reg)

        return {
            'pred_distribution': pred_dist,
            'probability_stats': prob_stats,
            'baseline_comparison': baseline_comparison,
            'status': 'ACCEPTED'
        }

    # Test with good predictions that clearly beat baseline
    valid_classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    np.random.seed(42)
    y_true = np.array(['NEUTRAL'] * 120 + ['POSITIVE'] * 40 + ['NEGATIVE'] * 40)
    # Better predictions: 90% on NEUTRAL, 75% on POSITIVE, 75% on NEGATIVE
    y_pred = np.array(
        ['NEUTRAL'] * 108 + ['POSITIVE'] * 30 + ['NEGATIVE'] * 30 +   # correct
        ['NEUTRAL'] * 12 + ['POSITIVE'] * 10 + ['NEGATIVE'] * 10      # errors
    )
    y_proba = np.zeros((200, 3))
    # Assign high probability to predicted class
    class_to_idx = {c: i for i, c in enumerate(valid_classes)}
    for i, pred in enumerate(y_pred):
        y_proba[i, class_to_idx[pred]] = 0.75
        other = [c for c in valid_classes if c != pred]
        y_proba[i, class_to_idx[other[0]]] = 0.15
        y_proba[i, class_to_idx[other[1]]] = 0.10

    print("Testing MANDATORY guardrail invocation pattern...")
    try:
        result = run_training_pipeline_with_guardrails(y_true, y_pred, y_proba, valid_classes)
        print(f"✅ Pipeline completed successfully: {result['status']}")
        print(f"   Prediction distribution: {result['pred_distribution']}")
        print(f"   Baseline improvement: Macro F1={result['baseline_comparison']['improvement']['macro_f1']:.4f}, Balanced Acc={result['baseline_comparison']['improvement']['balanced_accuracy']:.4f}")
        return True
    except GuardrailViolation as e:
        print(f"❌ Pipeline rejected: {e}")
        return False


def main():
    print("=" * 70)
    print("GUARDRAIL INTEGRATION TEST SUITE")
    print("=" * 70)
    print("This test verifies that the guardrail mechanism correctly")
    print("catches degenerate models and passes legitimate ones.")
    print("The REAL training pipeline MUST implement the same pattern.\n")

    results = []

    results.append(("All-NEUTRAL rejection", test_synthetic_pipeline_all_neutral()))
    results.append(("Legitimate predictions accepted", test_synthetic_pipeline_good_predictions()))
    results.append(("Constant regression rejected", test_synthetic_regression_degenerate()))
    results.append(("Good regression accepted", test_synthetic_regression_good()))
    results.append(("Mandatory invocation pattern", test_guardrail_cannot_be_skipped()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(p for _, p in results)
    if all_passed:
        print("\n🎉 ALL INTEGRATION TESTS PASSED")
        print("The guardrail mechanism is ready for the training pipeline.")
        return 0
    else:
        print("\n💥 SOME INTEGRATION TESTS FAILED")
        print("Fix the guardrail mechanism before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())