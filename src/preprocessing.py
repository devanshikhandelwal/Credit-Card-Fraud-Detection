
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CREDIT CARD FRAUD DETECTION
# FEATURE ENGINEERING
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# 1. HAVERSINE DISTANCE
# ==========================================

def calculate_distance(lat1, lon1, lat2, lon2):

    earth_radius = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius * c


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

def preprocess_data(df):

    df = df.copy()

    # Remove unnecessary index column
    df.drop(
        columns=["Unnamed: 0"],
        errors="ignore",
        inplace=True
    )

    # Convert date columns
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"]
    )

    df["dob"] = pd.to_datetime(df["dob"])

    # --------------------------------------
    # TIME FEATURES
    # --------------------------------------

    df["transaction_hour"] = (
        df["trans_date_trans_time"].dt.hour
    )

    df["transaction_day"] = (
        df["trans_date_trans_time"].dt.day
    )

    df["transaction_month"] = (
        df["trans_date_trans_time"].dt.month
    )

    df["day_of_week"] = (
        df["trans_date_trans_time"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------
    # CUSTOMER AGE
    # --------------------------------------

    df["customer_age"] = (
        (df["trans_date_trans_time"] - df["dob"])
        .dt.days / 365.25
    )

    # --------------------------------------
    # TRANSACTION AMOUNT
    # --------------------------------------

    df["log_amount"] = np.log1p(df["amt"])

    # --------------------------------------
    # CUSTOMER-MERCHANT DISTANCE
    # --------------------------------------

    df["distance_km"] = calculate_distance(
        df["lat"],
        df["long"],
        df["merch_lat"],
        df["merch_long"]
    )

    # --------------------------------------
    # SELECT USEFUL FEATURES
    # --------------------------------------

    selected_features = [
        "trans_date_trans_time",
        "amt",
        "log_amount",
        "category",
        "gender",
        "state",
        "job",
        "city_pop",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "day_of_week",
        "is_weekend",
        "customer_age",
        "distance_km",
        "is_fraud"
    ]

    df = df[selected_features]

    return df


# ==========================================
# 3. LOAD DATASETS
# ==========================================

print("Loading original datasets...")

train = pd.read_csv(DATA_DIR / "fraudTrain.csv")
test = pd.read_csv(DATA_DIR / "fraudTest.csv")

print("Training dataset:", train.shape)
print("Testing dataset:", test.shape)


# ==========================================
# 4. PREPROCESS BOTH DATASETS
# ==========================================

print("\nEngineering training features...")

train_processed = preprocess_data(train)

print("Engineering testing features...")

test_processed = preprocess_data(test)


# ==========================================
# 5. SAVE PROCESSED DATA
# ==========================================

train_processed.to_csv(
    OUTPUT_DIR / "train_processed.csv",
    index=False
)

test_processed.to_csv(
    OUTPUT_DIR / "test_processed.csv",
    index=False
)


# ==========================================
# 6. FINAL REPORT
# ==========================================

print("\n========== PREPROCESSING SUMMARY ==========")

print("Training shape:", train_processed.shape)
print("Testing shape:", test_processed.shape)

print("\nFeatures generated:")
print(train_processed.columns.tolist())

print("\nMissing values in training:")
print(train_processed.isnull().sum().sum())

print("\nFraud distribution:")
print(train_processed["is_fraud"].value_counts())

print("\nProcessed files saved successfully!")

print("\nPHASE 3 FEATURE ENGINEERING COMPLETED!")
