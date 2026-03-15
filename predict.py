"""
predict.py — Multi-city prediction utility with fuzzy location fallback.
Loaded by Flask routes. Do NOT run directly.
"""
import pickle
import numpy as np
import os
import json

MODEL_DIR = 'model'

def _load(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

model             = _load('best_model.pkl')
scaler            = _load('scaler.pkl')
columns           = _load('columns.pkl')
cities_locations  = _load('cities_locations.pkl')   # {city: [loc, ...]}


def get_cities():
    return sorted(cities_locations.keys()) if cities_locations else []


def get_locations(city=None):
    if not cities_locations:
        return []
    if city and city in cities_locations:
        return cities_locations[city]
    all_locs = []
    for locs in cities_locations.values():
        all_locs.extend(locs)
    return sorted(set(all_locs))


def get_cities_locations():
    return cities_locations or {}


def _resolve_location_key(city: str, location: str) -> tuple[str, str]:
    """
    Build and verify the location key used in one-hot encoding.
    Falls back progressively:
      1. Exact match:  'chennai__adyar'
      2. Strip spaces: 'chennai__adyar' (handles trailing/leading space)
      3. Prefix match: first column starting with 'location_key_chennai__ady'
      4. City 'other': 'location_key_chennai__other'

    Returns (resolved_key, match_type)
    """
    city     = city.lower().strip()
    location = location.lower().strip()

    # 1. Exact
    exact = f'location_key_{city}__{location}'
    if exact in columns:
        return exact, 'exact'

    # 2. Partial / prefix search (handles minor spacing issues)
    prefix = f'location_key_{city}__{location[:4]}'
    candidates = [c for c in columns if c.startswith(prefix)]
    if candidates:
        # Pick the closest candidate by string similarity
        best = min(candidates, key=lambda c: _edit_distance(
            c.replace(f'location_key_{city}__', ''), location
        ))
        return best, f'fuzzy→{best}'

    # 3. City-level other
    other_key = f'location_key_{city}__other'
    if other_key in columns:
        return other_key, 'other'

    # 4. Nothing matched — return empty string (all location cols stay 0)
    return '', 'unknown'


def _edit_distance(s1: str, s2: str) -> int:
    """Simple Levenshtein distance for fuzzy matching."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def predict_price(city: str, location: str, sqft: float, bhk: int, bath: int) -> dict:
    """
    Predict property price in Lakhs.

    Args:
        city     : 'bengaluru' or 'chennai'
        location : locality name
        sqft     : total square footage
        bhk      : number of bedrooms
        bath     : number of bathrooms

    Returns:
        dict with price, price_per_sqft, low, high, matched_location
    """
    if model is None or scaler is None or columns is None:
        raise RuntimeError(
            "Model not found. Please run 'python ml_model.py' first."
        )

    city     = city.lower().strip()
    location = location.lower().strip()

    x = np.zeros(len(columns))

    # Numeric features
    for col_name, val in [('total_sqft', sqft), ('bhk', bhk), ('bath', bath)]:
        if col_name in columns:
            x[columns.index(col_name)] = val

    # City one-hot
    city_col = f'city_{city}'
    if city_col in columns:
        x[columns.index(city_col)] = 1

    # Location one-hot — with fuzzy fallback
    loc_key, match_type = _resolve_location_key(city, location)
    if loc_key and loc_key in columns:
        x[columns.index(loc_key)] = 1

    x_scaled = scaler.transform([x])
    price = float(model.predict(x_scaled)[0])

    low  = round(price * 0.92, 2)
    high = round(price * 1.08, 2)
    pps  = round((price * 100_000) / sqft, 0) if sqft else 0

    # Matched display name
    if match_type == 'exact':
        display_loc = location
    elif match_type.startswith('fuzzy'):
        matched_raw = loc_key.replace(f'location_key_{city}__', '')
        display_loc = f'{location} (matched: {matched_raw})'
    elif match_type == 'other':
        display_loc = f'{location} (used area average)'
    else:
        display_loc = location

    return {
        'price':           round(price, 2),
        'price_per_sqft':  int(pps),
        'low':             low,
        'high':            high,
        'city':            city,
        'location':        location,
        'display_location': display_loc,
        'match_type':      match_type,
        'sqft':            sqft,
        'bhk':             bhk,
        'bath':            bath,
    }


def diagnose_location(city: str, location: str) -> dict:
    """
    Helper for debugging — returns full resolution info.
    Call from Flask shell: from predict import diagnose_location; print(diagnose_location('chennai','adyar'))
    """
    city     = city.lower().strip()
    location = location.lower().strip()

    city_locs    = cities_locations.get(city, []) if cities_locations else []
    in_locs_pkl  = location in city_locs
    loc_key, match = _resolve_location_key(city, location)
    key_in_cols  = loc_key in (columns or [])
    city_col     = f'city_{city}'
    city_in_cols = city_col in (columns or [])

    # Find any column that loosely contains the term
    partial_cols = [c for c in (columns or []) if location[:4] in c] if columns else []

    return {
        'city':              city,
        'location':          location,
        'in_locations_pkl':  in_locs_pkl,
        'resolved_key':      loc_key,
        'match_type':        match,
        'key_in_columns':    key_in_cols,
        'city_col_in_model': city_in_cols,
        'partial_column_matches': partial_cols[:10],
        'total_columns':     len(columns) if columns else 0,
    }


def get_model_metrics():
    path = os.path.join(MODEL_DIR, 'metrics.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def get_meta():
    path = os.path.join(MODEL_DIR, 'meta.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)