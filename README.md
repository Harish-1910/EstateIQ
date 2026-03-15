# EstateIQ — AI Real Estate Price Prediction

> ML-powered property valuation platform built with Flask.
> Supports Bengaluru, Chennai (Tamil Nadu), or **both datasets combined**.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Datasets](#2-datasets)
3. [Quick Start](#3-quick-start)
4. [Environment Variables](#4-environment-variables)
5. [ML Pipeline](#5-ml-pipeline)
6. [REST API](#6-rest-api)
7. [Running Tests](#7-running-tests)
8. [Docker Setup](#8-docker-setup)
9. [Deploying to Production](#9-deploying-to-production)
10. [Tech Stack](#10-tech-stack)

---

## 1. Project Structure

```
estateiq/
│
├── app.py                  Flask application factory
├── config.py               Configuration (DB URL, secret key, paths)
├── extensions.py           Flask extensions (db, bcrypt, login_manager)
├── models.py               SQLAlchemy models — User, Prediction
├── auth.py                 Blueprint — /login  /register  /logout
├── routes.py               Blueprint — /dashboard  /predict  /history  /api/*
│
├── ml_model.py             ★ RUN ONCE to train and save the ML model
├── predict.py              Prediction utility used by Flask routes
│
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── result.html
│   └── history.html
│
├── model/                  Auto-created after running ml_model.py
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── columns.pkl
│   ├── locations.pkl
│   ├── metrics.json
│   └── meta.json
│
├── requirements.txt
├── .env.example
├── .gitignore
├── Procfile
├── Dockerfile
├── docker-compose.yml
├── wsgi.py
├── tests.py
└── setup.cfg
```

---

## 2. Datasets

The model works with **one or both** datasets. Place whichever you have in the project root — the training script auto-detects them.

---

### Dataset 1 — Bengaluru House Price Data

```
https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data
```

1. Download and rename the file to:
```
Bengaluru_House_Data.csv
```
2. Place it in the project root.

**Columns used:**

| Column | Description | Common Schema |
|---|---|---|
| `location` | Locality name | → `area` |
| `total_sqft` | Area in sq. ft. | → `int_sqft` |
| `bath` | Bathrooms | → `n_bathroom` |
| `size` | e.g. "2 BHK" (parsed) | → `n_bedroom` |
| `price` | Sale price (already in Lakhs) | → `price_lakhs` |

---

### Dataset 2 — Chennai Housing Sales Price

```
https://www.kaggle.com/datasets/kunwarakash/chennai-housing-sales-price
```

1. Download and rename the file to:
```
Chennai_House_Price.csv
```
2. Place it in the project root.

**Columns used:**

| Column | Description | Common Schema |
|---|---|---|
| `AREA` | Locality name | → `area` |
| `INT_SQFT` | Interior area (sq. ft.) | → `int_sqft` |
| `N_BEDROOM` | Bedrooms | → `n_bedroom` |
| `N_BATHROOM` | Bathrooms | → `n_bathroom` |
| `N_ROOM` | Total rooms | → `n_room` |
| `DIST_MAINROAD` | Distance from main road | → `dist_mainroad` |
| `DATE_BUILD` | Build year (derives age) | → `property_age` |
| `SALES_PRICE` | Sale price in INR (/100,000) | → `price_lakhs` |

---

### Using both datasets together (recommended)

Place **both** CSV files in the project root. The training script will:

1. Load and clean each dataset independently
2. Normalise both to a **common schema** with identical column names
3. Add a `city` column (`bengaluru` / `chennai`) for each row
4. Concatenate them into one unified training set
5. Train a single model on the combined data

**Benefits of combining:**
- Larger training set → better generalisation
- Model learns cross-city price patterns
- More location diversity in predictions

**You can use just one dataset too** — the script skips whichever file is missing and trains on whatever is available.

---

### Other datasets (optional, same format)

| Name | Link |
|---|---|
| India House Price (multi-city) | https://www.kaggle.com/datasets/ankushpanday1/india-house-price-prediction |
| Real Estate — 7 Indian Cities | https://www.kaggle.com/datasets/rakkesharv/real-estate-data-from-7-indian-cities |
| Chennai alternate | https://www.kaggle.com/datasets/amaanafif/chennai-house-price |

---

## 3. Quick Start

### Step 1 — Install

```bash
git clone https://github.com/YOUR_USERNAME/estateiq.git
cd estateiq

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 2 — Configure

```bash
cp .env.example .env
# Edit .env → set SECRET_KEY and DATABASE_URL
```

### Step 3 — Place dataset(s) in project root

```
estateiq/
├── Bengaluru_House_Data.csv   ← one or both
├── Chennai_House_Price.csv    ← one or both
├── app.py
...
```

### Step 4 — Train

```bash
python ml_model.py
```

Sample output (both datasets):
```
------------------------------------------------------------
  Loading Bengaluru dataset: Bengaluru_House_Data.csv
------------------------------------------------------------
  Raw shape: (13320, 9)
  After cleaning: 9,892 rows

------------------------------------------------------------
  Loading Chennai dataset: Chennai_House_Price.csv
------------------------------------------------------------
  Raw shape: (7100, 22)
  After cleaning: 6,821 rows

------------------------------------------------------------
  Combining: Bengaluru + Chennai
------------------------------------------------------------
  Combined shape: 16,713 rows x 9 columns
  City breakdown:
    bengaluru     9,892 rows  (59.2%)
    chennai       6,821 rows  (40.8%)

  Model                     R2       MAE      RMSE     CV R2
  -----------------------------------------------------------------
  Linear Regression        0.7634    22.14    37.80    0.7501
  Ridge Regression         0.7701    21.87    37.21    0.7566
  Random Forest            0.9201    11.44    20.18    0.9043  <- best
  Gradient Boosting        0.9118    12.33    21.44    0.8974

  Best model: Random Forest  (R2 = 0.9201)
```

### Step 5 — Run

```bash
python app.py
```

Open **http://localhost:5000**

---

## 4. Environment Variables

```bash
# .env  (copy from .env.example — never commit this file)

# Generate a key: python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY=your-long-random-secret-key

# SQLite (development — no setup needed)
DATABASE_URL=sqlite:///realestate.db

# PostgreSQL (production)
# DATABASE_URL=postgresql://username:password@localhost:5432/estateiq_db

FLASK_ENV=development
FLASK_DEBUG=1
```

---

## 5. ML Pipeline

```
For each dataset present:
  Load CSV
  → Normalise column names
  → Drop unused columns
  → Clean string values and fix typos
  → Parse BHK from size string (Bengaluru only)
  → Derive PROPERTY_AGE from build year (Chennai only)
  → Fill missing values (median / mode)
  → Remove price outliers (1st–99th percentile per city)
  → Remove price-per-sqft outliers per locality
  → Map to common schema (area, int_sqft, n_bedroom, n_bathroom...)

Combine both cleaned dataframes
  → Add city one-hot column
  → Collapse localities with < 10 records → 'other'
  → Convert all prices to Lakhs
  → One-hot encode area + city columns
  → StandardScaler normalisation
  → 80/20 train/test split
  → Train 4 models with 5-fold cross-validation
  → Select best by R²
  → Save model/, scaler, columns, locations, metrics, meta
```

---

## 6. REST API

### POST /api/predict

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "anna nagar",
    "total_sqft": 1200,
    "bhk": 3,
    "bath": 2
  }'
```

Response:
```json
{
  "status": "success",
  "data": {
    "price": 72.50,
    "price_per_sqft": 6041,
    "low": 66.70,
    "high": 78.30,
    "location": "anna nagar",
    "sqft": 1200,
    "bhk": 3,
    "bath": 2
  }
}
```

### GET /api/locations

```bash
curl http://localhost:5000/api/locations
```

---

## 7. Running Tests

```bash
pip install pytest pytest-cov
pytest tests.py -v
pytest tests.py --cov=. --cov-report=term-missing
```

---

## 8. Docker Setup

```bash
# Start Flask + PostgreSQL
docker-compose up --build

# Stop
docker-compose down

# Reset DB
docker-compose down -v
```

---

## 9. Deploying to Production

### Render

1. Push to GitHub
2. New Web Service → connect repo
3. Set env vars: `SECRET_KEY`, `DATABASE_URL`
4. Start command: `gunicorn "app:create_app()" --bind 0.0.0.0:$PORT`

### Gunicorn (any Linux server)

```bash
gunicorn "app:create_app()" --workers 4 --bind 0.0.0.0:8000 --timeout 120
```

---

## 10. Tech Stack

| Layer | Tech |
|---|---|
| Language | Python 3.11 |
| Web Framework | Flask 3 |
| ORM / DB | SQLAlchemy + SQLite / PostgreSQL |
| Auth | Flask-Login + Flask-Bcrypt |
| ML | Scikit-learn (Random Forest, Gradient Boosting, Ridge, Linear) |
| Charts | Chart.js |
| Fonts | Cormorant Garamond + Syne |
| WSGI | Gunicorn |
| Containers | Docker + Docker Compose |
| CI | GitHub Actions |

---

## Licence

MIT — free to use, modify, and distribute.