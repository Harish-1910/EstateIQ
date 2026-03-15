"""
╔══════════════════════════════════════════════════════════════════════╗
║        EstateIQ — ML Training Script (Multi-City)                    ║
║        Bengaluru  +  Chennai (Tamil Nadu)                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Datasets required (place both in project root):                     ║
║                                                                      ║
║  1. Bengaluru_House_Data.csv                                         ║
║     https://www.kaggle.com/datasets/amitabhajoy/                     ║
║     bengaluru-house-price-data                                       ║
║                                                                      ║
║  2. Chennai_House_Price.csv                                          ║
║     https://www.kaggle.com/datasets/kunwarakash/                     ║
║     chennai-housing-sales-price                                      ║
║                                                                      ║
║  Run once:  python ml_model.py                                       ║
║  Output  :  model/  directory with all saved artefacts               ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════
#  PART 1 — LOAD & NORMALISE BENGALURU DATASET
#  Produces a clean DataFrame with columns:
#    city | location | total_sqft | bhk | bath | price_lakhs
# ═════════════════════════════════════════════════════════════════════
def load_bengaluru(path='Bengaluru_House_Data.csv'):
    print(f"  Loading Bengaluru → {path}")
    df = pd.read_csv(path)

    # Keep only useful columns
    df = df[['location', 'size', 'total_sqft', 'bath', 'price']].copy()
    df.dropna(inplace=True)

    # Parse BHK from size string (e.g. "3 BHK" → 3)
    def parse_bhk(s):
        try: return int(str(s).split()[0])
        except: return None

    df['bhk'] = df['size'].apply(parse_bhk)
    df.drop(columns=['size'], inplace=True)
    df.dropna(subset=['bhk'], inplace=True)
    df['bhk'] = df['bhk'].astype(int)
    df['bath'] = df['bath'].astype(int)

    # Parse sqft (handle ranges like "1000-1200" → average)
    def parse_sqft(s):
        try:
            s = str(s).strip()
            if '-' in s:
                parts = s.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            return float(s)
        except: return None

    df['total_sqft'] = df['total_sqft'].apply(parse_sqft)
    df.dropna(subset=['total_sqft'], inplace=True)

    # Clean location names
    df['location'] = df['location'].str.strip().str.lower()
    loc_counts = df['location'].value_counts()
    rare = loc_counts[loc_counts <= 5].index   # lowered: keeps smaller localities
    df['location'] = df['location'].apply(lambda x: 'other' if x in rare else x)

    # price is already in Lakhs in Bengaluru dataset
    df.rename(columns={'price': 'price_lakhs'}, inplace=True)

    # Price per sqft for outlier removal
    df['pps'] = df['price_lakhs'] * 100_000 / df['total_sqft']

    # Remove outliers: BHK vs sqft
    df = df[~(df['bhk'] > df['total_sqft'] / 300)]

    # Remove per-location pps outliers
    parts = []
    for loc, grp in df.groupby('location'):
        m, s = grp['pps'].mean(), grp['pps'].std()
        parts.append(grp if s == 0 else grp[(grp['pps'] > m-s) & (grp['pps'] < m+s)])
    df = pd.concat(parts, ignore_index=True)
    df.drop(columns=['pps'], inplace=True)

    # Bath outlier
    df = df[df['bath'] < df['bhk'] + 2]

    df['city'] = 'bengaluru'
    print(f"     {len(df):,} rows after cleaning")
    return df[['city', 'location', 'total_sqft', 'bhk', 'bath', 'price_lakhs']]


# ═════════════════════════════════════════════════════════════════════
#  PART 2 — LOAD & NORMALISE CHENNAI DATASET
#  Normalises to the same 6 columns:
#    city | location | total_sqft | bhk | bath | price_lakhs
# ═════════════════════════════════════════════════════════════════════
def load_chennai(path='Chennai_House_Price.csv'):
    print(f"  Loading Chennai   → {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.upper()

    # Drop irrelevant columns
    drop = ['PRT_ID', 'DATE_SALE', 'REG_FEE', 'COMMIS']
    df.drop(columns=[c for c in drop if c in df.columns], inplace=True)

    # Rename to unified schema
    rename_map = {
        'AREA':        'location',
        'INT_SQFT':    'total_sqft',
        'N_BEDROOM':   'bhk',
        'N_BATHROOM':  'bath',
        'SALES_PRICE': 'price_lakhs',
    }
    df.rename(columns=rename_map, inplace=True)

    # Keep only the core columns (drop all other Chennai-specific ones)
    required = ['location', 'total_sqft', 'bhk', 'bath', 'price_lakhs']
    df = df[[c for c in required if c in df.columns]].copy()
    df.dropna(subset=['price_lakhs'], inplace=True)

    # Convert INR → Lakhs
    df['price_lakhs'] = df['price_lakhs'] / 100_000

    # Clean location names
    df['location'] = df['location'].astype(str).str.strip().str.lower()
    area_fixes = {
        # Velachery variants
        'velchery': 'velachery', 'velcherry': 'velachery',
        # KK Nagar variants
        'kknagar': 'kk nagar', 'kk-nagar': 'kk nagar',
        # T Nagar variants
        'tnagar': 't nagar', 't-nagar': 't nagar',
        # Chrompet variants
        'chormpet': 'chrompet', 'chrompt': 'chrompet',
        'chrmpet': 'chrompet', 'chromepet': 'chrompet',
        # Anna Nagar variants
        'ana nagar': 'anna nagar', 'ann nagar': 'anna nagar',
        'annanagar': 'anna nagar',
        # Karapakkam variants
        'karapakam': 'karapakkam', 'karapakam': 'karapakkam',
        # Adyar variants  ← FIX: extra spellings added
        'adyr': 'adyar', 'adayar': 'adyar', 'aadyar': 'adyar',
        'adyar ': 'adyar',   # trailing space
        # Nungambakkam variants
        'nungampakam': 'nungambakkam', 'nungambakam': 'nungambakkam',
        'nungambakkam': 'nungambakkam',
    }
    df['location'] = df['location'].replace(area_fixes)

    # Numeric conversion
    for col in ['total_sqft', 'bhk', 'bath', 'price_lakhs']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df['bhk']  = df['bhk'].astype(int)
    df['bath'] = df['bath'].astype(int)

    # Collapse rare locations
    loc_counts = df['location'].value_counts()
    rare = loc_counts[loc_counts < 5].index   # lowered: keeps smaller areas like Adyar
    df['location'] = df['location'].apply(lambda x: 'other' if x in rare else x)
    if any('adyar' in str(r) for r in rare):
        print('    WARNING: adyar still has < 5 records in dataset')

    # Outlier removal: 1st–99th percentile on price
    p01, p99 = df['price_lakhs'].quantile(0.01), df['price_lakhs'].quantile(0.99)
    df = df[(df['price_lakhs'] >= p01) & (df['price_lakhs'] <= p99)]

    # Sqft sanity
    df = df[df['total_sqft'] > 100]

    # pps outliers per location
    df['pps'] = df['price_lakhs'] * 100_000 / df['total_sqft']
    parts = []
    for loc, grp in df.groupby('location'):
        m, s = grp['pps'].mean(), grp['pps'].std()
        parts.append(grp if s == 0 else grp[(grp['pps'] > m-s) & (grp['pps'] < m+s)])
    df = pd.concat(parts, ignore_index=True)
    df.drop(columns=['pps'], inplace=True)

    df['city'] = 'chennai'
    print(f"     {len(df):,} rows after cleaning")
    return df[['city', 'location', 'total_sqft', 'bhk', 'bath', 'price_lakhs']]


# ═════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════


if __name__ == '__main__':
    print("=" * 62)
    print("  EstateIQ  ·  Multi-City ML Training Pipeline")
    print("  Bengaluru  +  Chennai (Tamil Nadu)")
    print("=" * 62 + "\n")

    # ── Load available datasets (gracefully skip missing ones) ──────────
    frames = []
    missing = []

    # Supported Chennai filenames — handles typos in downloaded filename
    CHENNAI_CANDIDATES = [
        'Chennai_House_Price.csv',
        'Chennai_houseing_sale.csv',   # typo in Kaggle download filename
        'Chennai_Housing_Sale.csv',
        'chennai_house_price.csv',
    ]
    CHENNAI_FILE = next((f for f in CHENNAI_CANDIDATES if os.path.exists(f)), None)

    DATASET_PAIRS = [(load_bengaluru, 'Bengaluru_House_Data.csv')]
    if CHENNAI_FILE:
        DATASET_PAIRS.append((load_chennai, CHENNAI_FILE))
    else:
        print("  SKIPPING: No Chennai CSV found. Checked:")
        for c in CHENNAI_CANDIDATES:
            print(f"    - {c}")

    for loader, fname in DATASET_PAIRS:
        if os.path.exists(fname):
            try:
                frames.append(loader(fname))
            except Exception as e:
                print(f"  WARNING: Failed to load {fname}: {e}")
                missing.append(fname)
        else:
            print(f"  SKIPPING: {fname} not found")
            missing.append(fname)

    if not frames:
        raise RuntimeError(
            "\n  No datasets found! Please download at least one:\n"
            "  Bengaluru: https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data\n"
            "  Chennai  : https://www.kaggle.com/datasets/kunwarakash/chennai-housing-sales-price\n"
        )

    # ── Combine datasets ────────────────────────────────────────────────
    df = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined dataset: {len(df):,} rows")

    # Show city breakdown
    city_counts = df['city'].value_counts()
    for city, cnt in city_counts.items():
        print(f"    {city.capitalize():<15} {cnt:>6,} rows")

    # ── Create unique location keys with city prefix ────────────────────
    # e.g. "whitefield" in Bengaluru → "bengaluru__whitefield"
    #      "anna nagar"  in Chennai  → "chennai__anna nagar"
    df['location_key'] = df['city'] + '__' + df['location']

    # Save city→locations mapping for the UI dropdowns
    cities_locations = {}
    for city in df['city'].unique():
        locs = sorted(df[df['city'] == city]['location'].unique().tolist())
        cities_locations[city] = locs
        print(f"  {city.capitalize()} localities: {len(locs)}")

    with open(os.path.join(MODEL_DIR, 'cities_locations.pkl'), 'wb') as f:
        pickle.dump(cities_locations, f)

    # Flat list of all location keys (for predict.py)
    all_location_keys = sorted(df['location_key'].unique().tolist())
    with open(os.path.join(MODEL_DIR, 'locations.pkl'), 'wb') as f:
        pickle.dump(all_location_keys, f)

    # ── Feature Engineering ─────────────────────────────────────────────
    print("\n  Engineering features...")

    # One-hot encode: location_key + city
    df_enc = pd.get_dummies(df[['total_sqft', 'bhk', 'bath', 'location_key', 'city']])

    X = df_enc
    y = df['price_lakhs']

    columns = list(X.columns)
    with open(os.path.join(MODEL_DIR, 'columns.pkl'), 'wb') as f:
        pickle.dump(columns, f)
    print(f"  Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")

    # ── Scale ────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # ── Train / Test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ── Model Training ───────────────────────────────────────────────────
    print("\n  Training models...\n")

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression':  Ridge(alpha=10.0),
        'Random Forest':     RandomForestRegressor(n_estimators=150, max_depth=15,
                                                   random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,
                                                        max_depth=5, random_state=42),
    }

    results    = {}
    best_name  = None
    best_r2    = -999
    best_model = None
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"  {'Model':<25}  {'R2':>7}  {'MAE (L)':>10}  {'RMSE (L)':>10}  {'CV R2':>8}")
    print("  " + "-" * 68)

    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        cvs  = cross_val_score(mdl, X_scaled, y, cv=kf, scoring='r2')
        results[name] = {
            'MAE':        round(float(mae),        4),
            'RMSE':       round(float(rmse),       4),
            'R2':         round(float(r2),         4),
            'CV_R2_mean': round(float(cvs.mean()), 4),
            'CV_R2_std':  round(float(cvs.std()),  4),
        }
        flag = ' <- best' if r2 > best_r2 else ''
        print(f"  {name:<25}  {r2:>7.4f}  {mae:>10.2f}  {rmse:>10.2f}  "
              f"{cvs.mean():>7.4f}{flag}")
        if r2 > best_r2:
            best_r2, best_name, best_model = r2, name, mdl

    print(f"\n  Best model: {best_name}  (R2 = {best_r2:.4f})")

    # ── Save Artefacts ───────────────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)

    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump({'best_model': best_name, 'results': results}, f, indent=2)

    meta = {
        'cities':       list(city_counts.index),
        'city_counts':  city_counts.to_dict(),
        'total_rows':   int(len(df)),
        'price_unit':   'Lakhs (INR)',
        'missing_datasets': missing,
    }
    with open(os.path.join(MODEL_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n  Artefacts saved to model/:")
    for fname in sorted(os.listdir(MODEL_DIR)):
        size = os.path.getsize(os.path.join(MODEL_DIR, fname))
        print(f"    {fname:<28} {size/1024:>6.1f} KB")

    print(f"\n  Run:  python app.py")
    print("=" * 62)