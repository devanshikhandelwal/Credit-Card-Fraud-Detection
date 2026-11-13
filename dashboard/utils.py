
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# --------------------------------------------------
# Load saved model and preprocessing objects
# --------------------------------------------------

@staticmethod
def _unused():
    pass


@staticmethod
def _unused_2():
    pass


@staticmethod
def _unused_3():
    pass


@staticmethod
def _unused_4():
    pass


def load_model():
    model = joblib.load(MODEL_DIR / "xgboost_model.pkl")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    threshold_settings = joblib.load(
        MODEL_DIR / "threshold_settings.pkl"
    )

    return model, preprocessor, threshold_settings


# --------------------------------------------------
# Feature preparation
# --------------------------------------------------

def prepare_transaction(transaction):

    df = pd.DataFrame([transaction])

    # Calculate log transaction amount
    df["log_amount"] = np.log1p(df["amt"])

    # Convert date/time
    transaction_time = pd.to_datetime(
        df["trans_date_trans_time"]
    )

    df["transaction_hour"] = transaction_time.dt.hour
    df["transaction_day"] = transaction_time.dt.day
    df["transaction_month"] = transaction_time.dt.month
    df["day_of_week"] = transaction_time.dt.dayofweek
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Calculate customer age
    dob = pd.to_datetime(df["dob"])
    df["customer_age"] = (
        transaction_time.dt.year - dob.dt.year
        - (
            (transaction_time.dt.month < dob.dt.month)
            | (
                (transaction_time.dt.month == dob.dt.month)
                & (transaction_time.dt.day < dob.dt.day)
            )
        )
    )

    # Haversine distance between customer and merchant
    lat1 = np.radians(df["lat"].astype(float))
    lon1 = np.radians(df["long"].astype(float))
    lat2 = np.radians(df["merch_lat"].astype(float))
    lon2 = np.radians(df["merch_long"].astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    df["distance_km"] = (
        6371 * 2 * np.arcsin(np.sqrt(a))
    )

    # Select features in training order
    feature_columns = [
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
        "distance_km"
    ]

    return df[feature_columns]


# --------------------------------------------------
# Fraud prediction
# --------------------------------------------------

def predict_transaction(
    transaction,
    model,
    preprocessor,
    threshold
):

    features = prepare_transaction(transaction)

    transformed_features = preprocessor.transform(features)

    fraud_probability = model.predict_proba(
        transformed_features
    )[0][1]

    prediction = int(
        fraud_probability >= threshold
    )

    return {
        "fraud_probability": float(fraud_probability),
        "prediction": prediction,
        "status": (
            "Fraudulent"
            if prediction == 1
            else "Legitimate"
        )
    }
