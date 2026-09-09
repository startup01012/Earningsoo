#!/usr/bin/env python
"""
Dataset Immutability / Snapshot

Creates a reproducible snapshot of the current training dataset.
Records:
- File hash
- Shape
- Columns
- Manifest hash
- Row count
- Target counts
- Feature counts
- Creation timestamp

Outputs: data/processed/training_dataset_snapshot.json
"""

import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_df_hash(df: pd.DataFrame) -> str:
    """Compute hash of DataFrame content (shape + values)."""
    hasher = hashlib.sha256()
    hasher.update(str(df.shape).encode())
    hasher.update(','.join(df.columns).encode())
    sample = df.head(1000).to_numpy().tobytes()
    hasher.update(sample)
    return hasher.hexdigest()


def get_git_info():
    """Get git commit and status."""
    import subprocess
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd='/workspaces/EarningsOS'
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, cwd='/workspaces/EarningsOS'
        ).stdout.strip()
        return {
            "git_commit": commit if commit else "unknown",
            "git_status": "clean" if not status else "dirty",
            "git_dirty_files": status.split('\n') if status else []
        }
    except Exception as e:
        return {
            "git_commit": "unknown",
            "git_status": "unknown",
            "git_dirty_files": [str(e)]
        }


def create_snapshot():
    """Create dataset snapshot."""
    dataset_path = 'data/processed/earnings_ml_ready.parquet'
    manifest_path = 'data/processed/earnings_feature_manifest.csv'

    df = pd.read_parquet(dataset_path)
    manifest = pd.read_csv(manifest_path)

    git_info = get_git_info()

    # Column categorization
    identifier_cols = [
        'event_key', 'symbol', 'company_name', 'period_ended',
        'fiscal_quarter', 'result_announcement_datetime',
        'announcement_session_type', 'feature_cutoff_date',
        'reaction_start_date', 'quality_flags',
        'price_history_days', 'benchmark_history_days'
    ]
    audit_cols = [
        'fundamental_period_end_date', 'fundamental_estimated_available_date',
        'fundamental_pit_status', 'fundamental_lag_days'
    ]
    target_cols = manifest[manifest['is_target'] == True]['feature_name'].tolist()
    feature_cols = [c for c in df.columns if c not in identifier_cols and c not in audit_cols and c not in target_cols]

    # Verify counts
    total_count = len(identifier_cols) + len(audit_cols) + len(target_cols) + len(feature_cols)
    print(f"Column counts: identifiers={len(identifier_cols)}, audit={len(audit_cols)}, targets={len(target_cols)}, features={len(feature_cols)}, total={total_count}")

    # Feature groups
    feature_manifest = manifest[manifest['is_target'] == False]
    feature_groups = feature_manifest['feature_group'].value_counts().to_dict()

    # Target distribution
    target_dist = {}
    for t in target_cols:
        if t in df.columns:
            if df[t].dtype.kind in 'fc':
                target_dist[t] = {
                    "mean": float(df[t].mean()),
                    "std": float(df[t].std()),
                    "min": float(df[t].min()),
                    "max": float(df[t].max()),
                    "missing": int(df[t].isna().sum())
                }
            else:
                target_dist[t] = df[t].value_counts().to_dict()

    # Capture shape and counts BEFORE adding period_ended_dt column
    original_shape = {"rows": len(df), "columns": len(df.columns)}
    original_counts = {
        "identifiers": len(identifier_cols),
        "features": len(feature_cols),
        "targets": len(target_cols),
        "audit": len(audit_cols),
        "total": len(df.columns)
    }
    print(f"DEBUG - shape to save: {original_shape}")
    print(f"DEBUG - counts to save: {original_counts}")

    # Split info (adds period_ended_dt column)
    df['period_ended_dt'] = pd.to_datetime(df['period_ended'])
    train_max = pd.Timestamp('2021-12-31')
    val_max = pd.Timestamp('2024-03-31')

    train_mask = df['period_ended_dt'] <= train_max
    val_mask = (df['period_ended_dt'] > train_max) & (df['period_ended_dt'] <= val_max)
    test_mask = df['period_ended_dt'] > val_max

    split_info = {
        "method": "chronological_by_period_ended",
        "train": {"period_range": "2015-Q4 to 2021-Q4", "events": int(train_mask.sum())},
        "val": {"period_range": "2022-Q1 to 2024-Q1", "events": int(val_mask.sum())},
        "test": {"period_range": "2024-Q2 to 2026-Q2", "events": int(test_mask.sum())},
        "embargo_days": 5
    }

    # Reaction class distribution by split
    reaction_dist = {}
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        split_df = df[mask]
        reaction_dist[split_name] = split_df['reaction_class'].value_counts().to_dict()

    # Build snapshot using ORIGINAL shape/counts (before period_ended_dt added)
    debug_shape = original_shape
    debug_counts = original_counts
    print(f"DEBUG - shape to save: {debug_shape}")
    print(f"DEBUG - counts to save: {debug_counts}")
    print(f"DEBUG - shape to save: {debug_shape}")
    print(f"DEBUG - counts to save: {debug_counts}")

    snapshot = {
        "snapshot_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "dataset": {
            "path": dataset_path,
            "file_hash_sha256": compute_file_hash(dataset_path),
            "content_hash_sha256": compute_df_hash(df),
            "shape": debug_shape,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        },
        "manifest": {
            "path": manifest_path,
            "file_hash_sha256": compute_file_hash(manifest_path),
            "entries": len(manifest),
            "feature_groups": feature_groups
        },
        "columns": {
            "identifiers": identifier_cols,
            "features": feature_cols,
            "targets": target_cols,
            "audit": audit_cols,
            "counts": debug_counts
        },
        "splits": split_info,
        "reaction_class_distribution": reaction_dist,
        "target_statistics": target_dist,
        "git": get_git_info(),
        "limitations": {
            "fundamental_pit": "estimated_conservative (period_end + 45 days), not verified",
            "corporate_actions": "RAW_UNADJUSTED - no split/bonus/dividend adjustment",
            "benchmark": "NIFTYBEES ETF proxy, not official NIFTY 50/TRI",
            "consensus": "Unavailable",
            "survivorship_bias": "HIGH RISK - current NIFTY 50 constituents backfilled",
            "fundamental_coverage": "~37% (yfinance limitation ~2022+)"
        },
        "audit_files": {
            "feature_usability": "data/processed/feature_usability_audit.csv",
            "missingness": "data/processed/missingness_audit.csv",
            "missingness_strategy": "data/processed/missingness_strategy.md",
            "target_quality": "data/processed/target_quality_audit.csv",
            "temporal_split": "data/processed/temporal_split_audit.txt",
            "survivorship_bias": "data/processed/survivorship_bias_audit.txt",
            "price_integrity": "data/processed/price_integrity_audit.txt",
            "benchmark_integrity": "data/processed/benchmark_integrity_audit.txt",
            "announcement_events": "data/processed/announcement_event_audit.txt",
            "model_benchmark_spec": "data/processed/model_benchmark_spec.md",
            "model_acceptance_criteria": "data/processed/model_acceptance_criteria.md",
            "feature_ablation_spec": "data/processed/feature_ablation_spec.md",
            "quality_audit": "data/processed/earnings_quality_audit_report.txt"
        }
    }

    # Save snapshot
    snapshot_path = 'data/processed/training_dataset_snapshot.json'
    with open(snapshot_path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    # Read back to verify
    with open(snapshot_path, 'r') as f:
        saved = json.load(f)
    print(f"Saved shape: {saved['dataset']['shape']}")
    print(f"Saved column counts: {saved['columns']['counts']}")

    print(f"Snapshot created: {snapshot_path}")
    print(f"Dataset hash: {snapshot['dataset']['content_hash_sha256'][:16]}...")
    print(f"Manifest hash: {snapshot['manifest']['file_hash_sha256'][:16]}...")
    print(f"Git commit: {get_git_info()['git_commit'][:8]} ({get_git_info()['git_status']})")

    return snapshot


def verify_snapshot(snapshot_path: str = 'data/processed/training_dataset_snapshot.json') -> dict:
    """Verify current dataset matches snapshot."""
    with open(snapshot_path, 'r') as f:
        snapshot = json.load(f)

    df = pd.read_parquet(snapshot['dataset']['path'])
    manifest = pd.read_csv(snapshot['manifest']['path'])

    # Compute current column counts from snapshot's column lists
    identifier_cols = snapshot['columns']['identifiers']
    audit_cols = snapshot['columns']['audit']
    target_cols = snapshot['columns']['targets']
    feature_cols = snapshot['columns']['features']
    current_counts = {
        "identifiers": len(identifier_cols),
        "features": len(feature_cols),
        "targets": len(target_cols),
        "audit": len(audit_cols),
        "total": len(df.columns)
    }

    checks = {
        "dataset_file_hash": compute_file_hash(snapshot['dataset']['path']) == snapshot['dataset']['file_hash_sha256'],
        "manifest_file_hash": compute_file_hash(snapshot['manifest']['path']) == snapshot['manifest']['file_hash_sha256'],
        "dataset_shape": tuple(snapshot['dataset']['shape'].values()) == df.shape,
        "manifest_entries": snapshot['manifest']['entries'] == len(manifest),
        "column_counts": current_counts == snapshot['columns']['counts'],
        "git_commit_match": snapshot['git']['git_commit'] == get_git_info()['git_commit']
    }

    all_ok = all(checks.values())

    return {
        "verified": all_ok,
        "checks": checks,
        "snapshot_created_at": snapshot.get('created_at'),
        "current_git": get_git_info()['git_commit']
    }


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'verify':
        print("Verifying dataset snapshot...")
        result = verify_snapshot()
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0 if result['verified'] else 1)
    else:
        print("Creating dataset snapshot...")
        create_snapshot()
        print("\nVerifying snapshot...")
        result = verify_snapshot()
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()