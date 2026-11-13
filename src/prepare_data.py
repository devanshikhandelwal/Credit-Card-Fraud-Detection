
# ============================================
# CREDIT CARD FRAUD DETECTION
# PHASE 4: DATA PREPARATION
# ============================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import save_npz, csr_matrix

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================
# 2. LOAD PROCESSED DATA
# ============================================

print("Loading processed training data...")

df = pd.read_csv(
    DATA_DIR / "train_processed.csv",
    parse_dates=["trans_date_trans_time"]
)

print("Dataset shape:", df.shape)


# ============================================
# 3. CHRONOLOGICAL SORTING
# ============================================

df = df.sort_values(
    "trans_date_trans_time"
).reset_index(drop=True)

print("\nTransaction date range:")

print("Start:", df["trans_date_trans_time"].min())
print("End:", df["trans_date_trans_time"].max())


# ============================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================

X = df.drop(
    columns=["is_fraud", "trans_date_trans_time"]
)

y = df["is_fraud"]


# ============================================
# 5. DEFINE FEATURE TYPES
# ============================================

numeric_features = [
    "amt",
    "log_amount",
    "city_pop",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "day_of_week",
    "is_weekend",
    "customer_age",
    "distance_km"
]

categorical_features = [
    "category",
    "gender",
    "state",
    "job"
]


# ============================================
# 6. CHRONOLOGICAL TRAIN-VALIDATION SPLIT
# ============================================

split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index]
X_val = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_val = y.iloc[split_index:]

print("\n========== DATA SPLIT ==========")

print("Training records:", len(X_train))
print("Validation records:", len(X_val))

print(
    "Training date range:",
    df.iloc[:split_index]["trans_date_trans_time"].min(),
    "to",
    df.iloc[:split_index]["trans_date_trans_time"].max()
)

print(
    "Validation date range:",
    df.iloc[split_index:]["trans_date_trans_time"].min(),
    "to",
    df.iloc[split_index:]["trans_date_trans_time"].max()
)


# ============================================
# 7. CLASS DISTRIBUTION
# ============================================

print("\nTraining fraud distribution:")
print(y_train.value_counts())

print("\nValidation fraud distribution:")
print(y_val.value_counts())

print(
    "\nTraining fraud percentage:",
    round(y_train.mean() * 100, 4),
    "%"
)

print(
    "Validation fraud percentage:",
    round(y_val.mean() * 100, 4),
    "%"
)


# ============================================
# 8. ENCODING AND SCALING
# ============================================

print("\nCreating preprocessing pipeline...")

# Compatibility with different sklearn versions
try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=True
    )

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(with_mean=False),
            numeric_features
        ),
        (
            "categorical",
            encoder,
            categorical_features
        )
    ],
    remainder="drop",
    sparse_threshold=1.0
)


# ============================================
# 9. FIT ON TRAINING DATA ONLY
# ============================================

print("\nFitting encoder and scaler...")

X_train_processed = preprocessor.fit_transform(X_train)

print("Transforming validation data...")

X_val_processed = preprocessor.transform(X_val)

# Ensure sparse matrix format
X_train_processed = csr_matrix(X_train_processed)
X_val_processed = csr_matrix(X_val_processed)


# ============================================
# 10. SAVE PREPROCESSOR
# ============================================

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.pkl"
)


# ============================================
# 11. SAVE PROCESSED MATRICES
# ============================================

save_npz(
    OUTPUT_DIR / "X_train.npz",
    X_train_processed
)

save_npz(
    OUTPUT_DIR / "X_val.npz",
    X_val_processed
)

np.save(
    OUTPUT_DIR / "y_train.npy",
    y_train.to_numpy()
)

np.save(
    OUTPUT_DIR / "y_val.npy",
    y_val.to_numpy()
)


# ============================================
# 12. FINAL REPORT
# ============================================

print("\n========== PREPARATION SUMMARY ==========")

print("Training matrix:", X_train_processed.shape)
print("Validation matrix:", X_val_processed.shape)

print("Training matrix type:", type(X_train_processed))

print("\nPreprocessing pipeline saved!")

print("Training and validation data saved!")

print("\nPHASE 4 DATA PREPARATION COMPLETED!")
