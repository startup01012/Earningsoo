#!/usr/bin/env python
"""
Experiment Reproducibility Framework

Defines the metadata schema and utilities for tracking ML experiments.
This ensures every training run is fully reproducible.

Creates:
- experiments/ directory structure
- ExperimentManifest class for tracking
- Schema definition for experiment metadata
"""

import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


class ExperimentManifest:
    """
    Manages experiment metadata for reproducibility.

    Each experiment gets a unique ID and stores:
    - Git commit, dataset hash, feature manifest hash
    - Feature list, target, splits
    - Model name, hyperparameters, random seed
    - Software versions
    - Metrics and results
    """

    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.experiments_dir / "experiment_index.json"
        self._load_index()

    def _load_index(self):
        """Load experiment index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {"experiments": []}

    def _save_index(self):
        """Save experiment index."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    @staticmethod
    def get_git_commit() -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, cwd='/workspaces/EarningsOS'
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def get_git_status() -> str:
        """Get git status (clean/dirty)."""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True, text=True, cwd='/workspaces/EarningsOS'
            )
            return "clean" if result.stdout.strip() == "" else "dirty"
        except Exception:
            return "unknown"

    @staticmethod
    def get_file_hash(filepath: str) -> str:
        """Compute SHA256 hash of a file."""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def get_software_versions() -> Dict[str, str]:
        """Get key software versions."""
        import sklearn
        import lightgbm
        import xgboost
        import pandas
        import numpy

        versions = {
            "python": __import__('sys').version.split()[0],
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "xgboost": xgboost.__version__,
        }
        try:
            import catboost
            versions["catboost"] = catboost.__version__
        except ImportError:
            versions["catboost"] = "not_installed"
        return versions

    def create_experiment(
        self,
        experiment_name: str,
        dataset_path: str,
        feature_manifest_path: str,
        feature_list: List[str],
        target: str,
        feature_group: str,
        model_name: str,
        hyperparameters: Dict[str, Any],
        random_seed: int,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: str,
        test_end: str,
        embargo_days: int,
        preprocessing_version: str,
        description: str = ""
    ) -> str:
        """
        Create a new experiment entry.

        Returns experiment_id.
        """
        # Compute hashes
        dataset_hash = self.get_file_hash(dataset_path)
        manifest_hash = self.get_file_hash(feature_manifest_path)

        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(experiment_name.encode()).hexdigest()[:8]}"

        experiment = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "git_commit": self.get_git_commit(),
            "git_status": self.get_git_status(),
            "dataset_path": dataset_path,
            "dataset_hash": dataset_hash,
            "dataset_shape": self._get_dataset_shape(dataset_path),
            "feature_manifest_path": feature_manifest_path,
            "feature_manifest_hash": manifest_hash,
            "feature_list": feature_list,
            "feature_group": feature_group,
            "target": target,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
            "embargo_days": embargo_days,
            "preprocessing_version": preprocessing_version,
            "model_name": model_name,
            "hyperparameters": hyperparameters,
            "random_seed": random_seed,
            "software_versions": self.get_software_versions(),
            "description": description,
            "metrics": {},
            "status": "created"
        }

        # Save experiment detail
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        detail_path = exp_dir / "experiment_detail.json"
        with open(detail_path, 'w') as f:
            json.dump(experiment, f, indent=2)

        # Update index
        self.index["experiments"].append({
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "timestamp": experiment["timestamp"],
            "model_name": model_name,
            "target": target,
            "feature_group": feature_group,
            "status": "created"
        })
        self._save_index()

        print(f"Created experiment: {experiment_id}")
        return experiment_id

    def _get_dataset_shape(self, dataset_path: str) -> Dict[str, int]:
        """Get dataset shape."""
        df = pd.read_parquet(dataset_path)
        return {"rows": len(df), "columns": len(df.columns)}

    def update_metrics(self, experiment_id: str, metrics: Dict[str, Any], fold: Optional[int] = None):
        """Update experiment metrics."""
        exp_dir = self.experiments_dir / experiment_id
        detail_path = exp_dir / "experiment_detail.json"

        with open(detail_path, 'r') as f:
            experiment = json.load(f)

        if fold is not None:
            if "fold_metrics" not in experiment:
                experiment["fold_metrics"] = {}
            experiment["fold_metrics"][f"fold_{fold}"] = metrics
        else:
            experiment["metrics"] = metrics

        experiment["status"] = "completed" if not fold else "fold_completed"

        with open(detail_path, 'w') as f:
            json.dump(experiment, f, indent=2)

        # Update index
        for exp in self.index["experiments"]:
            if exp["experiment_id"] == experiment_id:
                exp["status"] = experiment["status"]
                exp["metrics"] = metrics
                break
        self._save_index()

    def add_predictions(self, experiment_id: str, split: str, predictions: pd.DataFrame):
        """Save predictions for an experiment."""
        exp_dir = self.experiments_dir / experiment_id
        pred_path = exp_dir / f"predictions_{split}.parquet"
        predictions.to_parquet(pred_path, index=False)

    def add_artifact(self, experiment_id: str, artifact_name: str, artifact_path: str):
        """Record an artifact (model, plot, etc.)."""
        exp_dir = self.experiments_dir / experiment_id
        detail_path = exp_dir / "experiment_detail.json"

        with open(detail_path, 'r') as f:
            experiment = json.load(f)

        if "artifacts" not in experiment:
            experiment["artifacts"] = {}

        experiment["artifacts"][artifact_name] = {
            "path": artifact_path,
            "hash": self.get_file_hash(artifact_path) if Path(artifact_path).exists() else None,
            "timestamp": datetime.now().isoformat()
        }

        with open(detail_path, 'w') as f:
            json.dump(experiment, f, indent=2)

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment details."""
        exp_dir = self.experiments_dir / experiment_id
        detail_path = exp_dir / "experiment_detail.json"
        if detail_path.exists():
            with open(detail_path, 'r') as f:
                return json.load(f)
        return None

    def list_experiments(self) -> pd.DataFrame:
        """List all experiments as DataFrame."""
        return pd.DataFrame(self.index["experiments"])

    def verify_reproducibility(self, experiment_id: str) -> Dict[str, Any]:
        """Verify that an experiment can be reproduced."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {"verified": False, "reason": "Experiment not found"}

        checks = {
            "git_commit_match": experiment["git_commit"] == self.get_git_commit(),
            "dataset_hash_match": experiment["dataset_hash"] == self.get_file_hash(experiment["dataset_path"]),
            "manifest_hash_match": experiment["feature_manifest_hash"] == self.get_file_hash(experiment["feature_manifest_path"]),
            "software_versions": experiment["software_versions"]
        }

        all_ok = all([
            checks["git_commit_match"],
            checks["dataset_hash_match"],
            checks["manifest_hash_match"]
        ])

        return {
            "verified": all_ok,
            "checks": checks,
            "experiment_id": experiment_id
        }


def main():
    """Demo the experiment manifest."""
    print("=" * 70)
    print("EXPERIMENT REPRODUCIBILITY FRAMEWORK - DEMO")
    print("=" * 70)

    manifest = ExperimentManifest("experiments")

    # Create a sample experiment
    exp_id = manifest.create_experiment(
        experiment_name="LightGBM_Variant_B",
        dataset_path="data/processed/earnings_ml_ready.parquet",
        feature_manifest_path="data/processed/earnings_feature_manifest.csv",
        feature_list=["return_1d", "return_5d", "volatility_20d", "prior_earnings_count"],
        target="reaction_class",
        feature_group="B",
        model_name="LightGBM",
        hyperparameters={
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        random_seed=42,
        train_start="2016-01-12",
        train_end="2021-12-31",
        val_start="2022-01-01",
        val_end="2024-03-31",
        test_start="2024-04-01",
        test_end="2026-06-30",
        embargo_days=5,
        preprocessing_version="v1.0",
        description="LightGBM with Market + Prior Earnings features (Variant B)"
    )

    # Simulate fold metrics
    manifest.update_metrics(exp_id, {"macro_f1": 0.28, "balanced_acc": 0.42}, fold=1)
    manifest.update_metrics(exp_id, {"macro_f1": 0.26, "balanced_acc": 0.40}, fold=2)
    manifest.update_metrics(exp_id, {"macro_f1": 0.27, "balanced_acc": 0.41}, fold=3)

    # Final metrics
    manifest.update_metrics(exp_id, {
        "macro_f1": 0.27,
        "balanced_accuracy": 0.41,
        "accuracy": 0.62,
        "log_loss": 0.85,
        "per_class_f1": {"NEGATIVE": 0.25, "NEUTRAL": 0.35, "POSITIVE": 0.21}
    })

    # List experiments
    print("\nExperiment Index:")
    print(manifest.list_experiments().to_string())

    # Verify reproducibility
    print("\nReproducibility Check:")
    verification = manifest.verify_reproducibility(exp_id)
    print(json.dumps(verification, indent=2))

    print("\n✅ Experiment framework ready!")
    print(f"Experiment directory: {manifest.experiments_dir}")


if __name__ == "__main__":
    main()