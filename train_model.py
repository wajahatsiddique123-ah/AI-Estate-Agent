"""
train_model.py   (FILE 1 of 3)
--------------------------------
Trains a K-Nearest-Neighbors regressor to predict Karachi property prices,
tunes k via GridSearchCV, evaluates it, and pickles everything the Streamlit
app needs to run predictions later:

    models/knn_model.pkl        -> trained KNeighborsRegressor
    models/scaler.pkl           -> fitted StandardScaler for numeric features
    models/encoders.pkl         -> fitted LabelEncoders for categorical features
    models/feature_columns.pkl  -> exact column order the model expects
    models/metrics.pkl          -> R2 / MAE / RMSE + predictions for charting

Run: python train_model.py
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH = "data/zameen_karachi_clean.csv"
MODEL_DIR = "models"

df = pd.read_csv(DATA_PATH)
df = df.dropna()  # safety net; dataset is already null-free

FEATURES_CAT = ["property_type", "location", "purpose"]
FEATURES_NUM = ["baths", "area_sqft", "bedrooms", "latitude", "longitude"]
TARGET = "price"

# Encode categoricals
encoders = {}
for col in FEATURES_CAT:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

feature_columns = [c + "_enc" for c in FEATURES_CAT] + FEATURES_NUM

X = df[feature_columns].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features (KNN is distance-based -> scaling is essential)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tune k
param_grid = {"n_neighbors": list(range(3, 26, 2)), "weights": ["uniform", "distance"]}
grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
print("Best params:", grid.best_params_)

y_pred = best_model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2:   {r2:.4f}")
print(f"MAE:  {mae:,.0f} PKR")
print(f"RMSE: {rmse:,.0f} PKR")

# Persist everything (pickle files)
import os
os.makedirs(MODEL_DIR, exist_ok=True)

with open(f"{MODEL_DIR}/knn_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(f"{MODEL_DIR}/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open(f"{MODEL_DIR}/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

metrics = {
    "r2": r2,
    "mae": mae,
    "rmse": rmse,
    "best_params": grid.best_params_,
    "y_test": y_test.values,
    "y_pred": y_pred,
    "cat_options": {c: sorted(df[c].unique().tolist()) for c in FEATURES_CAT},
    "num_ranges": {c: (float(df[c].min()), float(df[c].max())) for c in FEATURES_NUM},
}
with open(f"{MODEL_DIR}/metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print(f"\nSaved model + preprocessing artifacts to {MODEL_DIR}/")
