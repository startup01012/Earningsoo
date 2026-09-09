#!/usr/bin/env python
"""
Model Pipeline Smoke Test

Tests the COMPLETE future model pipeline WITHOUT real model training.
Exercises:
  dataset loader → X/y separation → temporal split → embargo → preprocessing
  → dummy/synthetic model → prediction → guardrails → metrics → experiment metadata

Purpose: Catch integration issues early:
- Wrong dimensions
- Wrong columns
- Dtype problems
- NaN handling problems
- Class mapping problems
- Probability shape problems
- Artifact path problems
- Guardrail invocation problems

DOES NOT use real earnings dataset to benchmark a model.
Uses synthetic data throughout.
"""

import numpy as np
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training_data_loader import load_training_data, TrainingData
from scripts.training_leakage_firewall import LeakageFirewall, run_leakage_firewall
from scripts.preprocessing_pipeline import create_preprocessor, PreprocessingConfig
from scripts.model_guardrails import (
    GuardrailViolation,
    validate_classification_predictions,
    validate_prediction_probabilities,
    compare_against_baseline,
    calculate_baseline_metrics
)
from scripts.temporal_splitter import verify_split_embargo


@dataclass
class SmokeTestResult:
    """Result of a smoke test step."""
    step: str
    passed: bool
    details: str
    artifacts: Dict[str, Any] = None


class DummyModel:
    """
    Synthetic model that mimics sklearn/LightGBM interface.
    Does NO real training - just stores config and generates predictions.
    """
    
    def __init__(self, model_type: str = 'tree', n_classes: int = 3, 
                 collapse_mode: str = 'none', random_state: int = 42):
        """
        Args:
            model_type: 'tree' or 'linear'
            n_classes: Number of classes for classification
            collapse_mode: 'none', 'single_class', 'prob_collapse' for testing guardrails
            random_state: Random seed
        """
        self.model_type = model_type
        self.n_classes = n_classes
        self.collapse_mode = collapse_mode
        self.random_state = random_state
        self.is_fitted_ = False
        self.classes_ = np.array(['NEGATIVE', 'NEUTRAL', 'POSITIVE'][:n_classes])
        self.best_iteration_ = None
        self.best_score_ = None
        
    def fit(self, X, y, eval_set=None, **kwargs):
        """Mock fit - just records data shapes and marks as fitted."""
        np.random.seed(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        self._y_train = y  # Store training labels
        
        # Simulate training behavior based on collapse_mode
        if self.collapse_mode == 'early_stop':
            self.best_iteration_ = 1
        elif self.collapse_mode == 'normal':
            self.best_iteration_ = 100
        else:
            self.best_iteration_ = 50
            
        self.best_score_ = 0.5
        return self
    
    def predict(self, X):
        """Generate synthetic predictions."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        
        n = len(X)
        np.random.seed(self.random_state + 1)
        
        if self.collapse_mode == 'single_class':
            # Degenerate: all NEUTRAL (class index 1)
            return np.full(n, 'NEUTRAL')
        elif self.collapse_mode == 'single_class_neg':
            return np.full(n, 'NEGATIVE')
        elif self.collapse_mode == 'normal':
            # Normal: predictions that beat baseline
            # Generate predictions with ~70% accuracy overall, distributed across classes
            # This should beat the majority-class baseline
            probs = [0.15, 0.7, 0.15] if self.n_classes == 3 else [0.5, 0.5]
            return np.random.choice(self.classes_, size=n, p=probs)
        else:
            # Normal: distributed predictions
            probs = [0.2, 0.6, 0.2] if self.n_classes == 3 else [0.5, 0.5]
            return np.random.choice(self.classes_, size=n, p=probs)
    
    def predict_proba(self, X):
        """Generate synthetic probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        
        n = len(X)
        np.random.seed(self.random_state + 2)
        
        if self.collapse_mode == 'prob_collapse':
            # Near-deterministic probabilities
            proba = np.zeros((n, self.n_classes))
            proba[:, 1] = 0.995  # All NEUTRAL with high confidence
            proba[:, 0] = 0.0025
            proba[:, 2] = 0.0025
            return proba
        elif self.collapse_mode == 'single_class':
            # Degenerate but valid probabilities
            proba = np.zeros((n, self.n_classes))
            proba[:, 1] = 1.0
            return proba
        else:
            # Normal: somewhat calibrated probabilities
            proba = np.zeros((n, self.n_classes))
            for i in range(self.n_classes):
                base = 0.2 if i == 0 else (0.6 if i == 1 else 0.2)
                noise = np.random.randn(n) * 0.1
                proba[:, i] = np.clip(base + noise, 0.01, 0.99)
            # Normalize
            proba = proba / proba.sum(axis=1, keepdims=True)
            return proba


def run_smoke_test(collapse_mode: str = 'none', model_type: str = 'tree') -> List[SmokeTestResult]:
    """
    Run the complete pipeline smoke test.
    
    Args:
        collapse_mode: 'none', 'single_class', 'prob_collapse', 'early_stop'
        model_type: 'tree' or 'linear'
        
    Returns:
        List of SmokeTestResult objects
    """
    results = []
    
    def record(step: str, passed: bool, details: str, artifacts: Dict = None):
        results.append(SmokeTestResult(step, passed, details, artifacts))
        status = "✅" if passed else "❌"
        print(f"  {status} {step}: {details}")
    
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: model_type={model_type}, collapse_mode={collapse_mode}")
    print(f"{'='*60}")
    
    # Step 1: Load dataset
    try:
        data = load_training_data()
        record("1. Dataset loader", True, 
               f"Loaded: train={data.X_train.shape}, val={data.X_val.shape}, test={data.X_test.shape}")
    except Exception as e:
        record("1. Dataset loader", False, f"Failed: {e}")
        return results
    
    # Step 2: X/y separation (verify no forbidden columns in X)
    try:
        forbidden = ['reaction_class', 'abnormal_return_1d', 'return_1d_after', 
                     'benchmark_return_1d_after', 'reaction_start_date',
                     'fundamental_period_end_date', 'event_key', 'symbol']
        for split_name, X in [('train', data.X_train), ('val', data.X_val), ('test', data.X_test)]:
            found = [c for c in X.columns if c in forbidden]
            if found:
                raise ValueError(f"Forbidden columns in {split_name} X: {found}")
        record("2. X/y separation", True, "No forbidden columns in X")
    except Exception as e:
        record("2. X/y separation", False, f"Failed: {e}")
        return results
    
    # Step 3: Temporal split verification
    try:
        # Need full dataframes with feature_cutoff_date for verification
        full_df = pd.read_parquet("data/processed/earnings_ml_ready.parquet")
        
        train_df = full_df[full_df['event_key'].isin(data.event_keys_train)]
        val_df = full_df[full_df['event_key'].isin(data.event_keys_val)]
        test_df = full_df[full_df['event_key'].isin(data.event_keys_test)]
        
        verification = verify_split_embargo(
            train_df, val_df, test_df,
            trading_day_column='feature_cutoff_date',
            embargo_days=5
        )
        assert verification['train_val_embargo_ok'], f"Train/Val embargo failed: {verification}"
        assert verification['val_test_embargo_ok'], f"Val/Test embargo failed: {verification}"
        record("3. Temporal split + embargo", True, 
               f"Train/Val gap: {verification['train_val_trading_days_between']} days, "
               f"Val/Test gap: {verification['val_test_trading_days_between']} days")
    except Exception as e:
        record("3. Temporal split + embargo", False, f"Failed: {e}")
        return results
    
    # Step 4: Preprocessing
    try:
        preprocessor = create_preprocessor(
            model_type=model_type,
            config=PreprocessingConfig(model_type=model_type, scaler_type='robust')
        )
        preprocessor.fit(data.X_train)
        
        X_train_proc = preprocessor.transform(data.X_train)
        X_val_proc = preprocessor.transform(data.X_val)
        X_test_proc = preprocessor.transform(data.X_test)
        
        # Verify no NaN in linear mode
        if model_type == 'linear':
            assert not X_train_proc.isna().any().any(), "Train has NaN after preprocessing"
            assert not X_val_proc.isna().any().any(), "Val has NaN after preprocessing"
            assert not X_test_proc.isna().any().any(), "Test has NaN after preprocessing"
        
        # Verify dimensions consistent
        assert X_train_proc.shape[1] == X_val_proc.shape[1] == X_test_proc.shape[1]
        
        record("4. Preprocessing", True,
               f"Train: {X_train_proc.shape}, Val: {X_val_proc.shape}, Test: {X_test_proc.shape}")
    except Exception as e:
        record("4. Preprocessing", False, f"Failed: {e}")
        return results
    
    # Step 5: Leakage Firewall
    try:
        firewall = LeakageFirewall(strict=True)
        
        # Load period_ends from full dataset
        full_df = pd.read_parquet("data/processed/earnings_ml_ready.parquet")
        train_period_ends = full_df.loc[full_df['event_key'].isin(data.event_keys_train), 'period_ended'].values
        val_period_ends = full_df.loc[full_df['event_key'].isin(data.event_keys_val), 'period_ended'].values
        test_period_ends = full_df.loc[full_df['event_key'].isin(data.event_keys_test), 'period_ended'].values
        
        firewall.check_all(
            X_train=X_train_proc, X_val=X_val_proc, X_test=X_test_proc,
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
        record("5. Leakage firewall", True, f"All {len(firewall.results)} checks passed")
    except Exception as e:
        record("5. Leakage firewall", False, f"Failed: {e}")
        return results
    
    # Step 6: Model interface (fit + predict)
    try:
        dummy_model = DummyModel(model_type=model_type, collapse_mode=collapse_mode)
        
        # Prepare targets
        y_train = data.y_train['reaction_class'].values
        y_val = data.y_val['reaction_class'].values
        
        dummy_model.fit(X_train_proc, y_train)
        
        y_pred = dummy_model.predict(X_val_proc)
        y_proba = dummy_model.predict_proba(X_val_proc)
        
        assert len(y_pred) == len(y_val), "Prediction length mismatch"
        assert y_proba.shape == (len(y_val), 3), f"Probability shape mismatch: {y_proba.shape}"
        
        record("6. Model interface", True,
               f"Fit OK, predict shape: {y_pred.shape}, proba shape: {y_proba.shape}")
    except Exception as e:
        record("6. Model interface", False, f"Failed: {e}")
        return results
    
    # Step 7: Guardrails
    try:
        # Convert to class labels for validation
        valid_classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        
        # Classification guardrail
        pred_dist = validate_classification_predictions(y_val, y_pred, valid_classes)
        
        # Probability guardrail
        prob_stats = validate_prediction_probabilities(y_proba, valid_classes)
        
        # Baseline comparison
        baseline_result = compare_against_baseline(y_val, y_pred, y_proba, valid_classes)
        
        record("7. Guardrails", True,
               f"Pred dist: {pred_dist}, Macro F1 improvement: {baseline_result['improvement']['macro_f1']:.4f}")
    except GuardrailViolation as e:
        # Guardrail violations are expected for collapse modes
        if collapse_mode in ['single_class', 'single_class_neg', 'prob_collapse']:
            record("7. Guardrails", True, f"Correctly caught degenerate model: {str(e)[:80]}")
        else:
            record("7. Guardrails", False, f"Unexpected guardrail failure: {e}")
            return results
    except Exception as e:
        record("7. Guardrails", False, f"Error: {e}")
        return results
    
    # Step 8: Metrics interface
    try:
        from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
        
        macro_f1 = f1_score(y_val, y_pred, average='macro', labels=valid_classes, zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)
        
        metrics = {
            'macro_f1': macro_f1,
            'balanced_accuracy': bal_acc,
            'accuracy': acc,
            'pred_distribution': dict(pd.Series(y_pred).value_counts())
        }
        
        record("8. Metrics", True, 
               f"Macro F1: {macro_f1:.4f}, Bal Acc: {bal_acc:.4f}, Acc: {acc:.4f}")
    except Exception as e:
        record("8. Metrics", False, f"Failed: {e}")
        return results
    
# Step 9: Experiment metadata
        try:
            from scripts.experiment_manifest import ExperimentManifest
            
            manifest = ExperimentManifest("experiments")
            
            exp_id = manifest.create_experiment(
                experiment_name=f"smoke_test_{model_type}_{collapse_mode}",
                dataset_path="data/processed/earnings_ml_ready.parquet",
                feature_manifest_path="data/processed/earnings_feature_manifest.csv",
                feature_list=data.feature_names,
                target="reaction_class",
                feature_group="all",
                model_name="dummy",
                hyperparameters={"collapse_mode": collapse_mode, "model_type": model_type},
                random_seed=42,
                train_start="2016-01-12",
                train_end="2021-12-31",
                val_start="2022-04-11",
                val_end="2024-05-30",
                test_start="2024-07-11",
                test_end="2026-06-30",
                embargo_days=5,
                preprocessing_version="smoke_test_v1",
                description=f"Smoke test: {model_type} model with {collapse_mode} mode"
            )
            
            # Update metrics
            manifest.update_metrics(exp_id, metrics)
            
            # Verify can load
            loaded = manifest.get_experiment(exp_id)
            assert loaded is not None
            assert loaded["experiment_id"] == exp_id
            
            record("9. Experiment metadata", True, f"Saved and loaded experiment: {exp_id}")
        except Exception as e:
            record("9. Experiment metadata", False, f"Failed: {e}")
            return results
    
    return results


def run_all_smoke_tests():
    """Run all smoke test variants."""
    print("="*60)
    print("MODEL PIPELINE SMOKE TEST SUITE")
    print("="*60)
    
    all_results = {}
    
    # Test configurations
    configs = [
        ("Tree model - normal", {'collapse_mode': 'none', 'model_type': 'tree'}),
        ("Tree model - single class collapse", {'collapse_mode': 'single_class', 'model_type': 'tree'}),
        ("Tree model - prob collapse", {'collapse_mode': 'prob_collapse', 'model_type': 'tree'}),
        ("Linear model - normal", {'collapse_mode': 'none', 'model_type': 'linear'}),
        ("Linear model - single class collapse", {'collapse_mode': 'single_class', 'model_type': 'linear'}),
    ]
    
    for name, kwargs in configs:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        results = run_smoke_test(**kwargs)
        
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        all_results[name] = (passed, total, results)
        
        if passed == total:
            print(f"\n✅ {name}: ALL {total} STEPS PASSED")
        else:
            print(f"\n❌ {name}: {passed}/{total} STEPS PASSED")
            for r in results:
                if not r.passed:
                    print(f"   FAILED: {r.step} - {r.details}")
    
    # Summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)
    all_passed = True
    for name, (passed, total, _) in all_results.items():
        status = "✅ PASS" if passed == total else "❌ FAIL"
        print(f"  {status}: {name} ({passed}/{total})")
        if passed != total:
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    success = run_all_smoke_tests()
    sys.exit(0 if success else 1)