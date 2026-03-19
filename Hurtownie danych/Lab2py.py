import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import os
import pyarrow.parquet as pq

# ====== Intel Accelerations ======
from sklearnex import patch_sklearn
patch_sklearn()

path = 'C:/Users/megaz/PycharmProjects/Lab-Studia-Semestr-3/data/yellow_tripdata_2025-01.parquet'
print("Sprawdzam czy plik istnieje...")

if not os.path.exists(path):
    raise FileNotFoundError(path)

print("Plik znaleziony.")

try:
    print("Wczytywanie pliku parquet...")
    table = pq.read_table(path, use_pandas_metadata=False)
except TypeError:
    print("Fallback dla starszego pyarrow...")
    table = pq.read_table(path)

df = table.to_pandas()
df = df.sample(100000, random_state=42)

print("Dataset wczytany.")
print("Rozmiar datasetu:", df.shape)
print("Kolumny:", df.columns.tolist())

# =========================
# FEATURE ENGINEERING
# =========================

print("\n--- FEATURE ENGINEERING ---")

df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

df["trip_duration_min"] = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60

df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

df["speed"] = df["trip_distance"] / (df["trip_duration_min"] / 60)

print("Dodano nowe kolumny:")
print(["trip_duration_min", "pickup_hour", "speed"])

# usunięcie oczywistych błędów
print("\n--- USUWANIE BŁĘDÓW ---")

before = len(df)

df = df[df["trip_duration_min"] > 0]
df = df[df["trip_distance"] > 0]
df = df[df["fare_amount"] > 0]

after = len(df)

print("Usunięto błędne rekordy:", before - after)
print("Nowy rozmiar:", df.shape)

# =========================
# USUWANIE OUTLIERÓW
# =========================

print("\n--- USUWANIE OUTLIERÓW ---")

for col in ["fare_amount", "trip_distance", "trip_duration_min"]:
    upper = df[col].quantile(0.99)
    before = len(df)
    df = df[df[col] < upper]
    after = len(df)

    print(f"{col} -> próg 99%: {upper:.2f}, usunięto:", before - after)

print("Rozmiar datasetu po usunięciu outlierów:", df.shape)

# =========================
# TARGET
# =========================

target = "fare_amount"

X = df.drop(columns=[target, "tpep_pickup_datetime", "tpep_dropoff_datetime"])
y = df[target]

print("\nFeatures:", X.columns.tolist())
print("Target:", target)

# =========================
# TRAIN TEST SPLIT
# =========================

print("\n--- TRAIN TEST SPLIT ---")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# =========================
# FEATURES
# =========================

numeric_features = [
    "trip_distance",
    "trip_duration_min",
    "speed"
]

categorical_features = [
    "VendorID",
    "payment_type",
    "pickup_hour"
]

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# =========================
# NUMERIC PIPELINE
# =========================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# =========================
# CATEGORICAL PIPELINE
# =========================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# =========================
# COLUMN TRANSFORMER
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# =========================
# MODEL + PIPELINE
# =========================

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nPipeline przygotowany.")

# =========================
# TRAINING
# =========================

print("\n--- TRENING MODELU ---")

pipeline.fit(X_train, y_train)

print("Model wytrenowany.")

# =========================
# PREDICTIONS
# =========================

print("\n--- PREDYKCJE ---")

y_pred = pipeline.predict(X_test)

print("Przykładowe predykcje:", y_pred[:10])

# =========================
# RMSE
# =========================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- WYNIK ---")
print("RMSE:", rmse)