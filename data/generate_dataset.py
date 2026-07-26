"""
generate_dataset.py
--------------------
Generates a synthetic Karachi real-estate dataset that mirrors the schema
of the real Kaggle dataset "Zameen.com Property Data Pakistan"
(https://www.kaggle.com/datasets/huzzefakhan/zameencom-property-data-pakistan).

WHY THIS EXISTS:
This sandbox has no internet access to pull the real Kaggle CSV. This script
builds a schema-identical, realistically-distributed stand-in so the rest of
the pipeline (EDA -> training -> pickle -> Streamlit app) can be built and
tested end to end today.

TO USE REAL DATA INSTEAD:
1. Download the CSV from Kaggle (link above) or the Karachi-specific dump.
2. Filter to city == 'Karachi'.
3. Save it as data/zameen_karachi.csv with the SAME column names used below.
4. Skip running this script — eda.py / train_model.py read that file directly.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 8000

# Real Karachi areas across price tiers (low -> high end)
areas = {
    "DHA Defence": (2.5, 1.0),
    "Clifton": (2.7, 1.0),
    "Bahria Town Karachi": (1.6, 0.6),
    "Gulshan-e-Iqbal Town": (1.1, 0.4),
    "PECHS": (1.4, 0.5),
    "North Nazimabad": (1.0, 0.35),
    "Gulistan-e-Jauhar": (0.9, 0.3),
    "Malir": (0.55, 0.2),
    "Korangi": (0.5, 0.18),
    "Nazimabad": (0.85, 0.3),
    "Federal B Area": (0.8, 0.28),
    "Scheme 33": (0.75, 0.25),
    "Shah Faisal Town": (0.6, 0.2),
    "Landhi": (0.45, 0.15),
    "Surjani Town": (0.5, 0.17),
}
area_names = list(areas.keys())

property_types = ["House", "Flat", "Upper Portion", "Lower Portion", "Room", "Farm House"]
purposes = ["For Sale", "For Rent"]
agencies = [f"Agency {i}" for i in range(1, 41)]
agents = [f"Agent {i}" for i in range(1, 121)]

rows = []
for i in range(N):
    loc = np.random.choice(area_names)
    base_mult, spread = areas[loc]

    ptype = np.random.choice(property_types, p=[0.32, 0.28, 0.14, 0.10, 0.06, 0.10])
    purpose = np.random.choice(purposes, p=[0.72, 0.28])

    if ptype == "Room":
        bedrooms = 1
        area_marla = np.random.uniform(1, 3)
    elif ptype == "Flat":
        bedrooms = np.random.choice([1, 2, 3, 4], p=[0.15, 0.4, 0.35, 0.1])
        area_marla = np.random.uniform(3, 12)
    elif ptype in ("Upper Portion", "Lower Portion"):
        bedrooms = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
        area_marla = np.random.uniform(4, 10)
    elif ptype == "Farm House":
        bedrooms = np.random.choice([4, 5, 6, 7], p=[0.3, 0.35, 0.2, 0.15])
        area_marla = np.random.uniform(40, 200)
    else:  # House
        bedrooms = np.random.choice([2, 3, 4, 5, 6], p=[0.15, 0.3, 0.3, 0.17, 0.08])
        area_marla = np.random.uniform(5, 40)

    baths = max(1, int(round(bedrooms * np.random.uniform(0.6, 1.1))))
    area_sqft = area_marla * 225  # 1 marla ~ 225 sqft

    # Price model: base per-marla rate for the area, scaled by size, bedrooms, noise
    per_marla_rate_lac = base_mult * np.random.normal(1.0, spread / max(base_mult, 0.1) * 0.15 + 0.08)
    per_marla_rate_lac = max(per_marla_rate_lac, 0.15)
    price_lac = per_marla_rate_lac * area_marla * np.random.uniform(0.9, 1.15)
    price_lac += bedrooms * np.random.uniform(1, 4)

    if purpose == "For Rent":
        # Rent is a small fraction of sale-equivalent value, expressed in PKR/month
        price = max(8000, (price_lac * 100000) * np.random.uniform(0.0035, 0.006))
    else:
        price = max(800000, price_lac * 100000)

    lat = 24.86 + np.random.uniform(-0.18, 0.18)
    lon = 67.01 + np.random.uniform(-0.18, 0.18)

    date_added = pd.Timestamp("2023-01-01") + pd.to_timedelta(np.random.randint(0, 700), unit="D")

    rows.append({
        "property_id": 100000 + i,
        "location_id": area_names.index(loc) + 1,
        "page_url": f"https://www.zameen.com/Property/karachi_{loc.lower().replace(' ', '_')}_{100000+i}.html",
        "property_type": ptype,
        "price": round(price, 0),
        "location": loc,
        "city": "Karachi",
        "province_name": "Sindh",
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "baths": baths,
        "area_sqft": round(area_sqft, 1),
        "purpose": purpose,
        "bedrooms": int(bedrooms),
        "date_added": date_added.strftime("%Y-%m-%d"),
        "agency": np.random.choice(agencies),
        "agent": np.random.choice(agents),
    })

df = pd.DataFrame(rows)

# Guarantee zero nulls
assert df.isnull().sum().sum() == 0, "Unexpected nulls in generated data"

out_path = "data/zameen_karachi.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df.isnull().sum())
print(df.head())
