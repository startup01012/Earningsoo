#!/usr/bin/env python
"""
Temporal Split / Walk-Forward Design Audit

Audits the existing chronological split and designs a proper walk-forward
evaluation protocol.

Key considerations:
- Split should be based on announcement_datetime or feature_cutoff_datetime
  rather than fiscal period_end alone
- Targets extend 1/3/5 trading days after announcement
- Need embargo period to prevent label contamination across folds

Outputs:
- data/processed/temporal_split_audit.csv
- data/processed/temporal_split_audit.txt
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path


def load_data():
    """Load ML-ready dataset."""
    df = pd.read_parquet('data/processed/earnings_ml_ready.parquet')
    return df


def analyze_current_split(df):
    """Analyze the current period_ended-based split."""
    df = df.copy()
    df['period_ended_dt'] = pd.to_datetime(df['period_ended'])
    df['announcement_dt'] = pd.to_datetime(df['result_announcement_datetime'])
    df['feature_cutoff_dt'] = pd.to_datetime(df['feature_cutoff_date'])
    df['reaction_start_dt'] = pd.to_datetime(df['reaction_start_date'])

    # Current split boundaries (from checkpoint 51)
    train_max_period = pd.Timestamp('2021-12-31')
    val_max_period = pd.Timestamp('2024-03-31')

    train_mask = df['period_ended_dt'] <= train_max_period
    val_mask = (df['period_ended_dt'] > train_max_period) & (df['period_ended_dt'] <= val_max_period)
    test_mask = df['period_ended_dt'] > val_max_period

    return train_mask, val_mask, test_mask


def check_announcement_vs_period_end(df):
    """Check relationship between announcement date and period end."""
    df = df.copy()
    df['period_ended_dt'] = pd.to_datetime(df['period_ended'])
    df['announcement_dt'] = pd.to_datetime(df['result_announcement_datetime'])

    # Days between period end and announcement
    df['days_to_announcement'] = (df['announcement_dt'] - df['period_ended_dt']).dt.days

    # Check for announcements BEFORE period end (advance/board events)
    early_ann = df[df['days_to_announcement'] < 0]
    late_ann = df[df['days_to_announcement'] > 90]

    return {
        'early_announcements': len(early_ann),
        'late_announcements': len(late_ann),
        'median_days_to_ann': df['days_to_announcement'].median(),
        'mean_days_to_ann': df['days_to_announcement'].mean(),
        'early_ann_details': early_ann[['event_key', 'symbol', 'period_ended', 'result_announcement_datetime', 'days_to_announcement']].to_dict('records'),
        'late_ann_details': late_ann[['event_key', 'symbol', 'period_ended', 'result_announcement_datetime', 'days_to_announcement']].to_dict('records'),
    }


def design_walkforward_splits(df):
    """Design walk-forward splits based on announcement dates."""
    df = df.copy()
    df['announcement_dt'] = pd.to_datetime(df['result_announcement_datetime'])
    df = df.sort_values('announcement_dt').reset_index(drop=True)

    # Get unique announcement dates (or quarters)
    announcements = df['announcement_dt'].dropna().sort_values().unique()

    # Design expanding window walk-forward
    # Fold 1: Train = first 60%, Val = next 20%
    # Fold 2: Train = first 70%, Val = next 10%
    # etc.

    n = len(announcements)
    fold_size = n // 5  # 5 folds

    folds = []
    for i in range(3):  # 3 folds minimum
        train_end = int(0.5 * n) + i * fold_size
        val_end = train_end + fold_size

        if val_end >= n:
            break

        train_ann = announcements[:train_end]
        val_ann = announcements[train_end:val_end]

        train_mask = df['announcement_dt'].isin(train_ann)
        val_mask = df['announcement_dt'].isin(val_ann)

        # Add embargo: exclude events within 5 trading days of val start
        val_start = val_ann[0]
        embargo_end = val_start + pd.Timedelta(days=7)  # ~5 trading days
        embargo_mask = (df['announcement_dt'] >= val_start) & (df['announcement_dt'] <= embargo_end)

        folds.append({
            'fold': i + 1,
            'train_start': train_ann[0],
            'train_end': train_ann[-1],
            'val_start': val_ann[0],
            'val_end': val_ann[-1],
            'train_events': int(train_mask.sum()),
            'val_events': int(val_mask.sum()),
            'embargo_events': int((train_mask & embargo_mask).sum()),
        })

    return folds


def check_label_window_contamination(df, train_mask, val_mask, test_mask):
    """Check if label windows cross split boundaries."""
    df = df.copy()
    df['announcement_dt'] = pd.to_datetime(df['result_announcement_datetime'])

    # Reaction windows: 1d, 3d, 5d after announcement
    # Approximate as calendar days (5 trading days ≈ 7 calendar days)
    df['label_window_end_1d'] = df['announcement_dt'] + pd.Timedelta(days=1)
    df['label_window_end_3d'] = df['announcement_dt'] + pd.Timedelta(days=4)
    df['label_window_end_5d'] = df['announcement_dt'] + pd.Timedelta(days=7)

    # Split boundaries
    train_end = pd.Timestamp('2021-12-31')
    val_end = pd.Timestamp('2024-03-31')

    results = []

    for split_name, mask, boundary in [('TRAIN', train_mask, train_end), ('VAL', val_mask, val_end)]:
        split_df = df[mask]
        for window in ['1d', '3d', '5d']:
            col = f'label_window_end_{window}'
            crosses = split_df[split_df[col] > boundary]
            results.append({
                'split': split_name,
                'label_window': f'{window}_after',
                'boundary': str(boundary.date()),
                'crossing_events': len(crosses),
                'pct': round(len(crosses) / len(split_df) * 100, 2)
            })

    return results


def check_feature_cutoff_vs_announcement(df):
    """Verify feature_cutoff is always before announcement."""
    df = df.copy()
    df['feature_cutoff_dt'] = pd.to_datetime(df['feature_cutoff_date'])
    df['announcement_dt'] = pd.to_datetime(df['result_announcement_datetime'])

    violations = df[df['feature_cutoff_dt'] > df['announcement_dt']]

    return {
        'violations': len(violations),
        'details': violations[['event_key', 'symbol', 'feature_cutoff_date', 'result_announcement_datetime']].to_dict('records') if len(violations) > 0 else []
    }


def generate_audit_report(df, ann_analysis, folds, label_contamination, cutoff_check):
    """Generate the audit report text."""
    txt = """======================================================================
TEMPORAL SPLIT / WALK-FORWARD DESIGN AUDIT
======================================================================

CURRENT SPLIT METHODOLOGY
-------------------------
Current split is based on fiscal period_ended (quarter end dates):
- Train: 2015-Q4 to 2021-Q4 (period_ended <= 2021-12-31)
- Val:   2022-Q1 to 2024-Q1 (period_ended 2022-03-31 to 2024-03-31)
- Test:  2024-Q2 to 2026-Q2 (period_ended > 2024-03-31)

ISSUE: Fiscal period end ≠ Information availability date
- A quarter ending June 30 may be announced in July/August
- Information becomes available at ANNOUNCEMENT time, not period end
- Current split uses period_ended which can misalign with information flow

ANNOUNCEMENT vs PERIOD END ANALYSIS
-----------------------------------
"""

    txt += f"Total events: {len(df)}\n"
    txt += f"Events announced BEFORE period end (advance/board events): {ann_analysis['early_announcements']}\n"
    txt += f"Events announced >90 days after period end: {ann_analysis['late_announcements']}\n"
    txt += f"Median days from period end to announcement: {ann_analysis['median_days_to_ann']:.1f}\n"
    txt += f"Mean days from period end to announcement: {ann_analysis['mean_days_to_ann']:.1f}\n\n"

    if ann_analysis['early_announcements'] > 0:
        txt += "EARLY ANNOUNCEMENTS (announcement < period_end):\n"
        for ev in ann_analysis['early_ann_details'][:10]:
            txt += f"  {ev['event_key']}: period_end={ev['period_ended']}, announced={ev['result_announcement_datetime']}, days_diff={ev['days_to_announcement']}\n"
        if len(ann_analysis['early_ann_details']) > 10:
            txt += f"  ... and {len(ann_analysis['early_ann_details']) - 10} more\n"
        txt += "\n"

    txt += "LABEL WINDOW CONTAMINATION CHECK\n"
    txt += "--------------------------------\n"
    txt += "Current split boundary: 2021-12-31 (Train/Val), 2024-03-31 (Val/Test)\n"
    txt += "Label windows extend 1/3/5 trading days AFTER announcement\n\n"

    for r in label_contamination:
        txt += f"  {r['split']} split, {r['label_window']} window: {r['crossing_events']} events ({r['pct']:.1f}%) cross boundary {r['boundary']}\n"

    txt += "\n"
    txt += "FEATURE CUTOFF vs ANNOUNCEMENT CHECK\n"
    txt += "-------------------------------------\n"
    txt += f"Violations (feature_cutoff > announcement): {cutoff_check['violations']}\n"
    if cutoff_check['violations'] > 0:
        for v in cutoff_check['details'][:5]:
            txt += f"  {v['event_key']}: cutoff={v['feature_cutoff_date']}, announcement={v['result_announcement_datetime']}\n"

    txt += "\n"
    txt += "======================================================================\n"
    txt += "PROPOSED WALK-FORWARD SPLIT DESIGN\n"
    txt += "======================================================================\n\n"

    txt += "PRINCIPLES:\n"
    txt += "1. Split by ANNOUNCEMENT date (information availability), not period_end\n"
    txt += "2. Use EXPANDING WINDOW (train grows, val slides forward)\n"
    txt += "3. Apply EMBARGO of 5 trading days (7 calendar days) between train and val\n"
    txt += "4. Test set held out as final unseen period\n\n"

    txt += "PROPOSED FOLDS:\n"
    txt += "| Fold | Train Period (announcement) | Val Period (announcement) | Train Events | Val Events | Embargo Events |\n"
    txt += "|------|-----------------------------|---------------------------|--------------|------------|----------------|\n"
    for fold in folds:
        txt += f"| {fold['fold']} | {fold['train_start'].date()} to {fold['train_end'].date()} | {fold['val_start'].date()} to {fold['val_end'].date()} | {fold['train_events']} | {fold['val_events']} | {fold['embargo_events']} |\n"

    txt += """

RECOMMENDED FINAL PROTOCOL:
---------------------------
1. PRIMARY SPLIT: Announcement-date based walk-forward (3-5 folds)
2. EMBARGO: 5 trading days (exclude train events within 5 days of val start)
3. TEST SET: Last 20% of events by announcement date (held out completely)
4. FEATURE CUTOFF: Must be strictly before announcement (already verified)
5. LABEL WINDOWS: 1d/3d/5d after announcement - ensure no cross-contamination

IMPLEMENTATION NOTES:
- Use announcement_dt for splitting, not period_ended
- Each fold's validation metrics are averaged for final performance
- Test set evaluated ONLY once after model selection
- No hyperparameter tuning on test set
"""

    return txt


def generate_csv_audit(df, ann_analysis, folds, label_contamination, cutoff_check):
    """Generate CSV audit data."""
    rows = []

    # Current split info
    rows.append({
        'audit_item': 'current_split_method',
        'detail': 'period_ended_based',
        'value': 'period_ended',
        'issue': 'Uses fiscal quarter end instead of announcement date'
    })

    # Announcement vs period end
    rows.append({
        'audit_item': 'announcement_vs_period_end',
        'detail': 'early_announcements',
        'value': ann_analysis['early_announcements'],
        'issue': 'Events announced before period end (advance/board events)'
    })
    rows.append({
        'audit_item': 'announcement_vs_period_end',
        'detail': 'late_announcements',
        'value': ann_analysis['late_announcements'],
        'issue': 'Events announced >90 days after period end'
    })
    rows.append({
        'audit_item': 'announcement_vs_period_end',
        'detail': 'median_days_to_announcement',
        'value': round(ann_analysis['median_days_to_ann'], 1),
        'issue': ''
    })

    # Label window contamination
    for r in label_contamination:
        rows.append({
            'audit_item': 'label_window_contamination',
            'detail': f"{r['split']}_{r['label_window']}",
            'value': r['crossing_events'],
            'issue': f"{r['pct']}% of {r['split']} events have label window crossing boundary"
        })

    # Feature cutoff check
    rows.append({
        'audit_item': 'feature_cutoff_check',
        'detail': 'violations',
        'value': cutoff_check['violations'],
        'issue': 'feature_cutoff > announcement (should be 0)'
    })

    # Proposed folds
    for fold in folds:
        rows.append({
            'audit_item': 'proposed_walkforward_fold',
            'detail': f"fold_{fold['fold']}",
            'value': f"train={fold['train_events']}_val={fold['val_events']}_embargo={fold['embargo_events']}",
            'issue': f"train: {fold['train_start'].date()} to {fold['train_end'].date()}, val: {fold['val_start'].date()} to {fold['val_end'].date()}"
        })

    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    df = load_data()

    print("Analyzing current split...")
    train_mask, val_mask, test_mask = analyze_current_split(df)

    print("Checking announcement vs period end...")
    ann_analysis = check_announcement_vs_period_end(df)

    print("Checking label window contamination...")
    label_contamination = check_label_window_contamination(df, train_mask, val_mask, test_mask)

    print("Checking feature cutoff vs announcement...")
    cutoff_check = check_feature_cutoff_vs_announcement(df)

    print("Designing walk-forward splits...")
    folds = design_walkforward_splits(df)

    print("Generating reports...")
    txt_report = generate_audit_report(df, ann_analysis, folds, label_contamination, cutoff_check)
    csv_df = generate_csv_audit(df, ann_analysis, folds, label_contamination, cutoff_check)

    # Save
    txt_path = 'data/processed/temporal_split_audit.txt'
    csv_path = 'data/processed/temporal_split_audit.csv'

    with open(txt_path, 'w') as f:
        f.write(txt_report)

    csv_df.to_csv(csv_path, index=False)

    print(f"Saved text report to {txt_path}")
    print(f"Saved CSV to {csv_path}")

    print("\n" + txt_report)

    return 0


if __name__ == "__main__":
    sys.exit(main())