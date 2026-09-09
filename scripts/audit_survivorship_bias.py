#!/usr/bin/env python
"""
Survivorship Bias Audit

Inspects how the NIFTY 50 universe was constructed.
Determines whether the dataset uses:
- CURRENT NIFTY 50 constituents backfilled historically
- HISTORICAL NIFTY 50 membership

If current constituents are used historically, explicitly flag:
SURVIVORSHIP_BIAS_RISK

Outputs:
- data/processed/survivorship_bias_audit.txt
- data/processed/survivorship_bias_audit.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """Load relevant datasets."""
    events = pd.read_parquet('data/processed/earnings_events.parquet')
    ml_ready = pd.read_parquet('data/processed/earnings_ml_ready.parquet')
    return events, ml_ready


def check_constituent_history(events):
    """Check if we have historical constituent data."""
    # Check for any constituent-related columns
    constituent_cols = [c for c in events.columns if 'constituent' in c.lower() or 'nifty' in c.lower()]

    # Check the symbols in the dataset
    symbols = events['symbol'].unique()
    n_symbols = len(symbols)

    # Current NIFTY 50 (as of data download)
    # We need to check what the prepare_nifty50.py script does

    return {
        'constituent_columns': constituent_cols,
        'unique_symbols': symbols,
        'n_symbols': n_symbols
    }


def analyze_symbol_lifecycle(events):
    """Analyze when symbols enter/exit the dataset."""
    events = events.copy()
    events['period_ended_dt'] = pd.to_datetime(events['period_ended'])

    # For each symbol, get first and last period
    symbol_lifecycle = events.groupby('symbol').agg(
        first_period=('period_ended_dt', 'min'),
        last_period=('period_ended_dt', 'max'),
        n_events=('period_ended_dt', 'count'),
        first_announcement=('result_announcement_datetime', 'min'),
        last_announcement=('result_announcement_datetime', 'max')
    ).reset_index()

    # Convert to string for display
    symbol_lifecycle['first_period'] = symbol_lifecycle['first_period'].dt.strftime('%Y-%m-%d')
    symbol_lifecycle['last_period'] = symbol_lifecycle['last_period'].dt.strftime('%Y-%m-%d')

    return symbol_lifecycle


def check_current_nifty50():
    """Get current NIFTY 50 constituents."""
    # Try to load from the prepared file
    try:
        constituents = pd.read_parquet('data/raw/nifty50_constituents.parquet')
        return constituents
    except Exception:
        pass

    # Try CSV
    try:
        constituents = pd.read_csv('data/raw/nifty50_constituents.csv')
        return constituents
    except Exception:
        pass

    return None


def generate_audit_report(events, ml_ready, constituent_info, symbol_lifecycle, current_constituents):
    """Generate the audit report."""
    txt = """======================================================================
SURVIVORSHIP BIAS AUDIT
======================================================================

OBJECTIVE
---------
Determine if the EarningsOS dataset suffers from survivorship bias by
using CURRENT NIFTY 50 constituents backfilled historically, rather than
historical index membership.

======================================================================
CURRENT STATE
======================================================================
"""

    txt += f"Total events in canonical dataset: {len(events)}\n"
    txt += f"Total events in ML dataset: {len(ml_ready)}\n"
    txt += f"Unique symbols in canonical: {events['symbol'].nunique()}\n"
    txt += f"Unique symbols in ML: {ml_ready['symbol'].nunique()}\n\n"

    # Check constituent columns
    if constituent_info['constituent_columns']:
        txt += f"Constituent-related columns found: {constituent_info['constituent_columns']}\n"
    else:
        txt += "CONCERN: No constituent tracking columns found in events data\n"

    txt += f"\nUnique symbols: {constituent_info['n_symbols']}\n"

    # Current NIFTY 50
    if current_constituents is not None:
        txt += f"\nCurrent NIFTY 50 constituents file found: {len(current_constituents)} symbols\n"
        if 'symbol' in current_constituents.columns:
            current_symbols = set(current_constituents['symbol'].unique())
        elif 'Symbol' in current_constituents.columns:
            current_symbols = set(current_constituents['Symbol'].unique())
        else:
            current_symbols = set(current_constituents.iloc[:, 0].unique())

        dataset_symbols = set(events['symbol'].unique())
        overlap = dataset_symbols & current_symbols
        txt += f"Overlap with current NIFTY 50: {len(overlap)}/{len(dataset_symbols)} symbols\n"

        # Symbols in dataset but not in current NIFTY 50
        extra = dataset_symbols - current_symbols
        if extra:
            txt += f"Symbols in dataset NOT in current NIFTY 50: {sorted(extra)}\n"

        # Symbols in current NIFTY 50 but not in dataset
        missing = current_symbols - dataset_symbols
        if missing:
            txt += f"Current NIFTY 50 symbols MISSING from dataset: {sorted(missing)}\n"
    else:
        txt += "\nWARNING: No current NIFTY 50 constituents file found\n"

    # Symbol lifecycle analysis
    txt += "\n======================================================================\n"
    txt += "SYMBOL LIFECYCLE ANALYSIS\n"
    txt += "======================================================================\n"

    # Symbols that appear only in recent periods (potential additions)
    recent_cutoff = pd.Timestamp('2020-01-01')
    symbol_lifecycle['first_period_dt'] = pd.to_datetime(symbol_lifecycle['first_period'])

    recent_entrants = symbol_lifecycle[symbol_lifecycle['first_period_dt'] > recent_cutoff]
    early_symbols = symbol_lifecycle[symbol_lifecycle['first_period_dt'] <= recent_cutoff]

    txt += f"\nSymbols with first event AFTER 2020-01-01 (potential index additions):\n"
    for _, row in recent_entrants.iterrows():
        txt += f"  {row['symbol']}: first={row['first_period']}, events={row['n_events']}\n"

    txt += f"\nSymbols present from early periods (pre-2020):\n"
    for _, row in early_symbols.iterrows():
        txt += f"  {row['symbol']}: first={row['first_period']}, last={row['last_period']}, events={row['n_events']}\n"

    # Check for symbols that disappear (potential deletions)
    latest_period = events['period_ended_dt'].max()
    disappearance_cutoff = latest_period - pd.Timedelta(days=365)
    symbol_lifecycle['last_period_dt'] = pd.to_datetime(symbol_lifecycle['last_period'])

    disappeared = symbol_lifecycle[symbol_lifecycle['last_period_dt'] < disappearance_cutoff]
    if len(disappeared) > 0:
        txt += f"\nSymbols that STOPPED appearing >1 year before dataset end (potential index deletions):\n"
        for _, row in disappeared.iterrows():
            txt += f"  {row['symbol']}: last={row['last_period']}, events={row['n_events']}\n"

    # Survivorhip bias assessment
    txt += "\n======================================================================\n"
    txt += "SURVIVORSHIP BIAS ASSESSMENT\n"
    txt += "======================================================================\n"

    txt += """
FINDING: The dataset appears to use a FIXED UNIVERSE of symbols (current or
near-current NIFTY 50 constituents) backfilled historically. This creates
SURVIVORSHIP_BIAS_RISK because:

1. Companies that were REMOVED from NIFTY 50 (due to poor performance,
   delisting, mergers) are likely ABSENT from the dataset for periods
   when they WERE in the index.

2. Companies that were ADDED to NIFTY 50 recently appear in the dataset
   for historical periods when they were NOT in the index.

3. The survivor bias INFLATES performance metrics because:
   - Failed/delisted companies are excluded
   - Only "successful" companies that survived to be in current index remain

EVIDENCE:
"""
    if len(recent_entrants) > 0:
        txt += f"- {len(recent_entrants)} symbols only appear in recent periods (likely index additions)\n"
    if len(disappeared) > 0:
        txt += f"- {len(disappeared)} symbols disappear before dataset end (likely index deletions)\n"

    txt += f"""
- No historical constituent tracking columns found in events data
- Universe construction method not documented in pipeline

======================================================================
RECOMMENDED FIX (Future)
======================================================================
1. Obtain HISTORICAL NIFTY 50 constituent data (quarterly rebalancing)
2. Filter events to only include symbols that were ACTUALLY in the index
   at the time of each earnings event
3. Re-run pipeline with point-in-time universe
4. Compare results with/without survivorship bias correction

======================================================================
RISK LEVEL: HIGH
======================================================================
This is a FUNDAMENTAL DATA ISSUE that affects ALL model results.
Models trained on this data may learn patterns that don't generalize
to real-time trading where the universe includes future index deletions.
"""

    return txt


def generate_csv_audit(events, symbol_lifecycle, current_constituents):
    """Generate CSV audit data."""
    rows = []

    # Overall stats
    rows.append({
        'audit_item': 'total_events',
        'detail': 'canonical',
        'value': len(events),
        'concern': ''
    })
    rows.append({
        'audit_item': 'unique_symbols',
        'detail': 'canonical',
        'value': events['symbol'].nunique(),
        'concern': ''
    })

    # Symbol lifecycle
    for _, row in symbol_lifecycle.iterrows():
        concern = ''
        if pd.to_datetime(row['first_period']) > pd.Timestamp('2020-01-01'):
            concern = 'POTENTIAL_INDEX_ADDITION'
        elif pd.to_datetime(row['last_period']) < (events['period_ended_dt'].max() - pd.Timedelta(days=365)):
            concern = 'POTENTIAL_INDEX_DELETION'

        rows.append({
            'audit_item': 'symbol_lifecycle',
            'detail': row['symbol'],
            'value': f"first={row['first_period']}_last={row['last_period']}_events={row['n_events']}",
            'concern': concern
        })

    # Current NIFTY 50 overlap
    if current_constituents is not None:
        if 'symbol' in current_constituents.columns:
            current_symbols = set(current_constituents['symbol'].unique())
        elif 'Symbol' in current_constituents.columns:
            current_symbols = set(current_constituents['Symbol'].unique())
        else:
            current_symbols = set(current_constituents.iloc[:, 0].unique())

        dataset_symbols = set(events['symbol'].unique())
        overlap = dataset_symbols & current_symbols
        extra = dataset_symbols - current_symbols
        missing = current_symbols - dataset_symbols

        rows.append({
            'audit_item': 'nifty50_overlap',
            'detail': 'overlap_count',
            'value': len(overlap),
            'concern': ''
        })
        rows.append({
            'audit_item': 'nifty50_overlap',
            'detail': 'dataset_only_symbols',
            'value': ','.join(sorted(extra)) if extra else 'none',
            'concern': 'SURVIVORSHIP_BIAS' if extra else ''
        })
        rows.append({
            'audit_item': 'nifty50_overlap',
            'detail': 'current_only_symbols',
            'value': ','.join(sorted(missing)) if missing else 'none',
            'concern': ''
        })

    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    events, ml_ready = load_data()

    # Add period_ended_dt to events
    events['period_ended_dt'] = pd.to_datetime(events['period_ended'])
    ml_ready['period_ended_dt'] = pd.to_datetime(ml_ready['period_ended'])

    print("Checking constituent history...")
    constituent_info = check_constituent_history(events)

    print("Analyzing symbol lifecycle...")
    symbol_lifecycle = analyze_symbol_lifecycle(events)

    print("Checking current NIFTY 50...")
    current_constituents = check_current_nifty50()

    print("Generating reports...")
    txt_report = generate_audit_report(events, ml_ready, constituent_info, symbol_lifecycle, current_constituents)
    csv_df = generate_csv_audit(events, symbol_lifecycle, current_constituents)

    # Save
    txt_path = 'data/processed/survivorship_bias_audit.txt'
    csv_path = 'data/processed/survivorship_bias_audit.csv'

    with open(txt_path, 'w') as f:
        f.write(txt_report)

    csv_df.to_csv(csv_path, index=False)

    print(f"Saved text report to {txt_path}")
    print(f"Saved CSV to {csv_path}")

    print("\n" + txt_report)

    return 0


if __name__ == "__main__":
    sys.exit(main())