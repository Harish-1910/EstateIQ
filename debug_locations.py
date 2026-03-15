"""
debug_locations.py
==================
Run this BEFORE ml_model.py to inspect your dataset and find
exactly why a location like 'adyar' might be missing or broken.

Usage:
    python debug_locations.py
"""
import pandas as pd
import numpy as np
import os, sys

SEP = "=" * 60

def check_dataset(path, location_col, price_col, search_term="adyar"):
    print(f"\n{SEP}")
    print(f"  Checking: {path}")
    print(SEP)

    if not os.path.exists(path):
        print(f"  SKIP — file not found: {path}")
        return

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.upper()

    if location_col not in df.columns:
        print(f"  ERROR — column '{location_col}' not found")
        print(f"  Available columns: {list(df.columns)}")
        return

    # Raw values
    raw = df[location_col].astype(str).str.strip().str.lower()
    print(f"\n  Total rows: {len(df):,}")
    print(f"\n  RAW values containing '{search_term}':")
    hits = raw[raw.str.contains(search_term, case=False, na=False)]
    if len(hits):
        print(hits.value_counts().to_string())
    else:
        print(f"  NONE found. Closest matches:")
        close = raw[raw.str.contains(search_term[:3], case=False, na=False)]
        print(close.value_counts().head(10).to_string() if len(close) else "  None")

    # Full count
    counts = raw.value_counts()
    print(f"\n  Locations with < 10 records (will collapse to 'other'):")
    rare = counts[counts < 10]
    if search_term in rare.index:
        print(f"  *** '{search_term}' has only {rare[search_term]} records — WILL BE COLLAPSED ***")
    print(f"  {len(rare)} locations with < 10 records out of {len(counts)} total")

    print(f"\n  Top 30 locations by record count:")
    print(counts.head(30).to_string())


# ── Check both datasets ────────────────────────────────────────────
check_dataset('Chennai_House_Price.csv',   'AREA',     'SALES_PRICE', 'adyar')
check_dataset('Bengaluru_House_Data.csv',  'LOCATION', 'PRICE',       'adyar')

# ── Check saved model columns ─────────────────────────────────────
print(f"\n{SEP}")
print("  Checking saved model columns (model/columns.pkl)")
print(SEP)

import pickle
cols_path = 'model/columns.pkl'
locs_path = 'model/cities_locations.pkl'

if os.path.exists(cols_path):
    with open(cols_path, 'rb') as f:
        columns = pickle.load(f)
    adyar_cols = [c for c in columns if 'adyar' in c.lower()]
    print(f"\n  Columns containing 'adyar': {adyar_cols if adyar_cols else 'NONE — not in model!'}")
    print(f"  Total feature columns: {len(columns)}")
else:
    print("  model/columns.pkl not found — run ml_model.py first")

if os.path.exists(locs_path):
    with open(locs_path, 'rb') as f:
        cl = pickle.load(f)
    for city, locs in cl.items():
        adyar_locs = [l for l in locs if 'adyar' in l.lower()]
        print(f"\n  {city} locations containing 'adyar': {adyar_locs if adyar_locs else 'NONE'}")
        print(f"  Total {city} locations: {len(locs)}")
else:
    print("  model/cities_locations.pkl not found — run ml_model.py first")

print(f"\n{SEP}")
print("  Done. Read the output above to find the root cause.")
print(SEP)