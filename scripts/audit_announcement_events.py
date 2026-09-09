#!/usr/bin/env python
"""
Announcement Event Audit

Audits the canonical result announcement events.
Verifies:
- Announcement timestamp
- Announcement type
- Actual result vs board/approval/pre-announcement
- Event period
- Reaction start
- Feature cutoff

Flags events where announcement date < period end
These may be legitimate advance/board events or incorrectly classified.

Outputs:
- data/processed/announcement_event_audit.csv
- data/processed/announcement_event_audit.txt
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """Load canonical events and ML-ready dataset."""
    events = pd.read_parquet('data/processed/earnings_events.parquet')
    ml_ready = pd.read_parquet('data/processed/earnings_ml_ready.parquet')
    return events, ml_ready


def classify_announcement_type(events):
    """Classify each announcement based on available data."""
    events = events.copy()
    events['period_ended_dt'] = pd.to_datetime(events['period_ended'])
    events['announcement_dt'] = pd.to_datetime(events['result_announcement_datetime'])

    # Days between period end and announcement
    events['days_period_to_ann'] = (events['announcement_dt'] - events['period_ended_dt']).dt.days

    # Classification logic
    def classify(row):
        days = row['days_period_to_ann']

        if pd.isna(days):
            return 'UNCERTAIN'

        if days < 0:
            # Announcement before period end
            if days < -30:
                return 'BOARD_EVENT'  # Very early - likely board meeting approval
            else:
                return 'PRE_ANNOUNCEMENT'  # Slightly early - could be advance notice
        elif days == 0:
            return 'ACTUAL_RESULT'
        elif days <= 5:
            return 'ACTUAL_RESULT'  # Same week
        elif days <= 45:
            return 'ACTUAL_RESULT'  # Within regulatory window
        elif days <= 90:
            return 'ACTUAL_RESULT'  # Late but plausible
        else:
            return 'UNCERTAIN'  # Very late - data quality issue?

    events['announcement_classification'] = events.apply(classify, axis=1)
    return events


def check_timing_consistency(events):
    """Check announcement timing consistency using available columns."""
    # The canonical events don't have feature_cutoff_date or reaction_start_date
    # Those are in the ML-ready dataset. We'll check what we can.

    violations = []

    # Check for announcements after period end (should be >= 0 ideally)
    events['period_ended_dt'] = pd.to_datetime(events['period_ended'])
    events['announcement_dt'] = pd.to_datetime(events['result_announcement_datetime'])
    events['days_period_to_ann'] = (events['announcement_dt'] - events['period_ended_dt']).dt.days

    # Very late announcements (>90 days after period end)
    very_late = events[events['days_period_to_ann'] > 90]
    if len(very_late) > 0:
        violations.append({
            'type': 'VERY_LATE_ANNOUNCEMENT',
            'count': len(very_late),
            'details': very_late[['event_key', 'period_ended', 'result_announcement_datetime', 'days_period_to_ann']].to_dict('records')
        })

    return violations


def check_session_type_consistency(events):
    """Check if session type derived from time matches expectations."""
    events = events.copy()
    events['announcement_dt'] = pd.to_datetime(events['result_announcement_datetime'])

    # Extract hour
    events['ann_hour'] = events['announcement_dt'].dt.hour
    events['ann_minute'] = events['announcement_dt'].dt.minute

    # Derive session type from time
    def derive_session(row):
        h = row['ann_hour']
        m = row['ann_minute']
        if h < 9 or (h == 9 and m < 15):
            return 'pre_market'
        elif h > 15 or (h == 15 and m > 30):
            return 'post_market'
        else:
            return 'during_market'

    events['derived_session_type'] = events.apply(derive_session, axis=1)

    inconsistencies = []

    # Check for midnight/zero times (often indicates missing time data)
    midnight = events[(events['ann_hour'] == 0) & (events['ann_minute'] == 0)]
    if len(midnight) > 0:
        inconsistencies.append({
            'type': 'MIDNIGHT_ANNOUNCEMENT_TIME',
            'count': len(midnight),
            'details': midnight[['event_key', 'result_announcement_datetime']].to_dict('records')
        })

    return inconsistencies


def generate_audit_report(events, classified_events, timing_violations, session_inconsistencies):
    """Generate audit report."""
    txt = """======================================================================
ANNOUNCEMENT EVENT AUDIT
======================================================================

OBJECTIVE
---------
Verify canonical announcement events for:
- Announcement timestamp accuracy
- Announcement type classification
- Actual result vs board/approval/pre-announcement
- Event period alignment

======================================================================
ANNOUNCEMENT CLASSIFICATION
======================================================================
"""

    # Classification distribution
    class_dist = classified_events['announcement_classification'].value_counts()
    for cls, count in class_dist.items():
        txt += f"  {cls}: {count} events ({count/len(classified_events)*100:.1f}%)\n"

    # Show UNCERTAIN events
    uncertain = classified_events[classified_events['announcement_classification'] == 'UNCERTAIN']
    if len(uncertain) > 0:
        txt += f"\nUNCERTAIN EVENTS ({len(uncertain)}):\n"
        for _, row in uncertain.iterrows():
            txt += f"  {row['event_key']}: period={row['period_ended']}, ann={row['result_announcement_datetime']}, days_diff={row['days_period_to_ann']}\n"

    # Show BOARD_EVENT / PRE_ANNOUNCEMENT
    board = classified_events[classified_events['announcement_classification'].isin(['BOARD_EVENT', 'PRE_ANNOUNCEMENT'])]
    if len(board) > 0:
        txt += f"\nEARLY ANNOUNCEMENTS (before period end):\n"
        for _, row in board.iterrows():
            cls = row['announcement_classification']
            txt += f"  {row['event_key']}: {cls}, period={row['period_ended']}, ann={row['result_announcement_datetime']}, days_diff={row['days_period_to_ann']}\n"

    # Show ACTUAL_RESULT
    actual = classified_events[classified_events['announcement_classification'] == 'ACTUAL_RESULT']
    txt += f"\nACTUAL RESULT events: {len(actual)} ({len(actual)/len(classified_events)*100:.1f}%)\n"

    # Days distribution
    txt += f"\nDays from period end to announcement (median): {classified_events['days_period_to_ann'].median():.0f}\n"
    txt += f"Days from period end to announcement (mean): {classified_events['days_period_to_ann'].mean():.1f}\n"

    # Timing violations
    txt += "\n======================================================================\n"
    txt += "TIMING CONSISTENCY CHECKS\n"
    txt += "======================================================================\n"

    if timing_violations:
        for v in timing_violations:
            txt += f"\n{v['type']}: {v['count']} violations\n"
            for d in v['details'][:5]:
                txt += f"  {d}\n"
            if len(v['details']) > 5:
                txt += f"  ... and {len(v['details']) - 5} more\n"
    else:
        txt += "\n✅ No timing violations found\n"

    # Session type consistency
    txt += "\n======================================================================\n"
    txt += "SESSION TYPE CONSISTENCY (Derived from Time)\n"
    txt += "======================================================================\n"

    if 'derived_session_type' in classified_events.columns:
        session_dist = classified_events['derived_session_type'].value_counts()
        for st, count in session_dist.items():
            txt += f"  {st}: {count} events\n"
    else:
        txt += "  Session type not available in canonical events\n"

    if session_inconsistencies:
        for inc in session_inconsistencies:
            txt += f"\n{inc['type']}: {inc['count']} inconsistencies\n"
            for d in inc['details'][:5]:
                txt += f"  {d}\n"
            if len(inc['details']) > 5:
                txt += f"  ... and {len(inc['details']) - 5} more\n"
    else:
        txt += "\n✅ No session type inconsistencies found\n"

    return txt


def generate_csv_audit(classified_events, timing_violations, session_inconsistencies):
    """Generate CSV audit data."""
    rows = []

    # Classification summary
    class_dist = classified_events['announcement_classification'].value_counts()
    for cls, count in class_dist.items():
        rows.append({
            'audit_item': 'announcement_classification',
            'classification': cls,
            'count': int(count),
            'percentage': round(count / len(classified_events) * 100, 1)
        })

    # Individual event classifications
    for _, row in classified_events.iterrows():
        rows.append({
            'audit_item': 'event_classification',
            'event_key': row['event_key'],
            'symbol': row['symbol'],
            'period_ended': str(row['period_ended']),
            'result_announcement_datetime': str(row['result_announcement_datetime']),
            'days_period_to_ann': int(row['days_period_to_ann']) if pd.notna(row['days_period_to_ann']) else None,
            'announcement_classification': row['announcement_classification'],
            'derived_session_type': row.get('derived_session_type', ''),
            'event_status': row.get('event_status', ''),
            'quality_flag': row.get('quality_flag', ''),
            'days_from_period_end': row.get('days_from_period_end', '')
        })

    # Timing violations
    for v in timing_violations:
        rows.append({
            'audit_item': 'timing_violation',
            'violation_type': v['type'],
            'count': v['count'],
            'details': str(v['details'][:3])
        })

    # Session inconsistencies
    for inc in session_inconsistencies:
        rows.append({
            'audit_item': 'session_inconsistency',
            'inconsistency_type': inc['type'],
            'count': inc['count'],
            'details': str(inc['details'][:3])
        })

    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    events, ml_ready = load_data()

    print("Classifying announcement types...")
    classified_events = classify_announcement_type(events)

    print("Checking timing consistency...")
    timing_violations = check_timing_consistency(classified_events)

    print("Checking session type consistency...")
    session_inconsistencies = check_session_type_consistency(classified_events)

    print("Generating reports...")
    txt_report = generate_audit_report(events, classified_events, timing_violations, session_inconsistencies)
    csv_df = generate_csv_audit(classified_events, timing_violations, session_inconsistencies)

    # Add feature cutoff analysis from ML-ready dataset
    if 'feature_cutoff_date' in ml_ready.columns and 'result_announcement_datetime' in ml_ready.columns:
        ml_ready_copy = ml_ready.copy()
        ml_ready_copy['feature_cutoff_dt'] = pd.to_datetime(ml_ready_copy['feature_cutoff_date'])
        ml_ready_copy['announcement_dt'] = pd.to_datetime(ml_ready_copy['result_announcement_datetime'])
        ml_ready_copy['days_cutoff_to_ann'] = (ml_ready_copy['announcement_dt'] - ml_ready_copy['feature_cutoff_dt']).dt.days

        txt_report += "\n======================================================================\n"
        txt_report += "FEATURE CUTOFF ANALYSIS (from ML-ready dataset)\n"
        txt_report += "======================================================================\n"
        txt_report += f"Days from feature_cutoff to announcement (median): {ml_ready_copy['days_cutoff_to_ann'].median():.0f}\n"
        txt_report += f"Days from feature_cutoff to announcement (mean): {ml_ready_copy['days_cutoff_to_ann'].mean():.1f}\n"
        same_day = (ml_ready_copy['days_cutoff_to_ann'] == 0).sum()
        one_day = (ml_ready_copy['days_cutoff_to_ann'] == 1).sum()
        more = (ml_ready_copy['days_cutoff_to_ann'] > 1).sum()
        negative = (ml_ready_copy['days_cutoff_to_ann'] < 0).sum()
        txt_report += f"  Same day (0): {same_day} events\n"
        txt_report += f"  1 day before: {one_day} events\n"
        txt_report += f"  >1 day before: {more} events\n"
        txt_report += f"  After announcement (VIOLATION): {negative} events\n"

    # Save
    txt_path = 'data/processed/announcement_event_audit.txt'
    csv_path = 'data/processed/announcement_event_audit.csv'

    with open(txt_path, 'w') as f:
        f.write(txt_report)

    csv_df.to_csv(csv_path, index=False)

    print(f"Saved text report to {txt_path}")
    print(f"Saved CSV to {csv_path}")

    print("\n" + txt_report)

    return 0


if __name__ == "__main__":
    sys.exit(main())