# Class Imbalance Strategy for EarningsOS

## Current Class Distribution

The `reaction_class` target has the following approximate distribution:

| Class | Count | Percentage |
|-------|-------|------------|
| NEUTRAL | ~1,144 | ~60% |
| POSITIVE | ~384 | ~20% |
| NEGATIVE | ~364 | ~19% |

This is a moderately imbalanced classification problem (3:1 ratio for majority vs minority).

## Previous Failure Mode

The previous LightGBM training failure exhibited:
- Model predicted NEUTRAL for ~95%+ of validation samples
- Early stopping at iteration 1
- Probabilities collapsed to near-deterministic NEUTRAL
- Pipeline reported "training completed" without detecting degenerate predictions

This was a classic case of the model exploiting class imbalance by predicting the majority class.

## Defined Future Experiments

### Experiment 1: Standard Loss (Baseline)
- **Loss function**: Standard cross-entropy / multiclass logloss
- **Class weights**: None (uniform)
- **Purpose**: Establish baseline behavior; confirm guardrails catch collapse

### Experiment 2: Class-Weighted Loss
- **Loss function**: Weighted cross-entropy
- **Class weights**: `n_samples / (n_classes * n_samples_per_class)` (balanced)
  - NEUTRAL: 1.0
  - POSITIVE: ~3.0
  - NEGATIVE: ~3.1
- **Alternative**: `class_weight='balanced'` in LightGBM/XGBoost
- **Purpose**: Test if weighting mitigates majority-class collapse

### Experiment 3: Focal Loss (Future)
- **Loss function**: Focal loss with gamma > 0
- **Purpose**: Focus training on hard-to-classify examples
- **Status**: Not implemented yet; requires custom loss

## What We Will NOT Do

| Technique | Reason |
|-----------|--------|
| Random oversampling (SMOTE, ADASYN) | Breaks temporal structure; creates synthetic events that didn't occur |
| Undersampling majority class | Discards valuable historical data; reduces statistical power |
| Threshold tuning on validation | Data leakage; must be part of nested CV or separate threshold optimization |

## Evaluation Metrics

**Primary Metrics (Classification):**
1. **Macro F1** - Equally weights all classes; main guardrail metric
2. **Balanced Accuracy** - Average of per-class recall; main guardrail metric
3. **Per-class F1** - Positive F1, Negative F1 (NEUTRAL F1 also tracked)
4. **Per-class Precision** - Positive Precision, Negative Precision
5. **Per-class Recall** - Positive Recall, Negative Recall

**Metrics Explicitly DEMOTED:**
- **Accuracy** - Misleading with imbalanced classes (60% NEUTRAL baseline)
- **Micro F1** - Dominated by majority class
- **Weighted F1** - Similar issues to accuracy

## Guardrail Integration

The model guardrails (`scripts/model_guardrails.py`) enforce:

1. **Minimum unique classes**: At least 2 classes must be predicted
2. **Baseline comparison**: Model must beat majority-class baseline by:
   - Macro F1 improvement ≥ 0.01
   - Balanced Accuracy improvement ≥ 0.01
3. **Probability collapse detection**: Flags if >95% samples have max prob > 0.99

These guardrails are MANDATORY and run after EVERY training run.

## Threshold Selection

If probability calibration is needed for downstream use:

1. **Do NOT** tune thresholds on validation set directly
2. **Use** nested cross-validation within training set
3. **Or** use a separate held-out calibration set (not val/test)
4. **Default**: Use argmax (0.5 threshold for binary, multinomial for 3-class)

## Future Considerations

- If class-weighted loss shows promise, test `scale_pos_weight` equivalent for multiclass
- Consider cost-sensitive learning if economic costs of errors are asymmetric
- Monitor per-class performance drift in walk-forward evaluation
- If NEUTRAL remains dominant, consider whether 3-class is the right formulation
  - Alternative: Binary (REACTION vs NO_REACTION) with NEUTRAL as NO_REACTION
  - Alternative: Regression on abnormal_return_3d with threshold at inference

## Decision Rule for Model Acceptance

A model is ACCEPTED for further evaluation ONLY if:
1. ✅ Macro F1 > Baseline Macro F1 + 0.01
2. ✅ Balanced Accuracy > Baseline Balanced Accuracy + 0.01
3. ✅ At least 2 unique classes predicted
4. ✅ No probability collapse (max prob < 0.99 for >95% samples)
5. ✅ Per-class F1 > 0 for all classes (no zero-F1 classes)

If ANY condition fails → **MODEL REJECTED** (guardrail violation)

## Documentation of Results

For each experiment, record:
- Experiment ID and configuration
- Class weights used (if any)
- Per-class metrics (F1, Precision, Recall)
- Macro F1 and Balanced Accuracy
- Guardrail pass/fail status
- Best iteration and early stopping behavior
- Predicted class distribution on validation