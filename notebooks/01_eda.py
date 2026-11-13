# ============================================
# CREDIT CARD FRAUD DETECTION
# PHASE 2: EXPLORATORY DATA ANALYSIS
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------

print("Loading datasets...")

train = pd.read_csv(DATA_DIR / "fraudTrain.csv")

print("\nDataset loaded successfully!")

print("Training shape:", train.shape)

# --------------------------------------------
# 3. BASIC DATA INFORMATION
# --------------------------------------------

print("\n========== DATASET INFORMATION ==========")

print("\nColumn names:")
print(train.columns.tolist())

print("\nData types:")
print(train.dtypes)

print("\nMissing values:")
print(train.isnull().sum().sum())

print("\nDuplicate rows:")
print(train.duplicated().sum())

# Remove unnecessary index column
if "Unnamed: 0" in train.columns:
    train.drop(columns=["Unnamed: 0"], inplace=True)

# --------------------------------------------
# 4. CLASS DISTRIBUTION
# --------------------------------------------

print("\n========== FRAUD DISTRIBUTION ==========")

fraud_counts = train["is_fraud"].value_counts()

print(fraud_counts)

fraud_percentage = (
    train["is_fraud"].value_counts(normalize=True) * 100
)

print("\nPercentage distribution:")
print(fraud_percentage)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=train,
    x="is_fraud"
)

plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.savefig(
    OUTPUT_DIR / "class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------
# 5. TRANSACTION AMOUNT ANALYSIS
# --------------------------------------------

print("\n========== TRANSACTION AMOUNT ==========")

print(train["amt"].describe())

plt.figure(figsize=(10, 5))

sns.histplot(
    data=train,
    x="amt",
    hue="is_fraud",
    bins=50,
    log_scale=(False, True),
    element="step",
    stat="density",
    common_norm=False
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Density")

plt.savefig(
    OUTPUT_DIR / "transaction_amount.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------
# 6. FRAUD BY TRANSACTION CATEGORY
# --------------------------------------------

category_fraud = (
    train.groupby("category")["is_fraud"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

category_fraud["fraud_percentage"] = (
    category_fraud["mean"] * 100
)

print("\n========== FRAUD BY CATEGORY ==========")

print(category_fraud)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=category_fraud.reset_index(),
    x="category",
    y="fraud_percentage"
)

plt.title("Fraud Percentage by Transaction Category")
plt.xlabel("Transaction Category")
plt.ylabel("Fraud Percentage")

plt.xticks(rotation=45, ha="right")

plt.savefig(
    OUTPUT_DIR / "fraud_by_category.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------
# 7. TRANSACTION TIME ANALYSIS
# --------------------------------------------

train["trans_date_trans_time"] = pd.to_datetime(
    train["trans_date_trans_time"]
)

train["transaction_hour"] = (
    train["trans_date_trans_time"].dt.hour
)

hourly_fraud = (
    train.groupby("transaction_hour")["is_fraud"]
    .mean() * 100
)

plt.figure(figsize=(12, 5))

sns.lineplot(
    x=hourly_fraud.index,
    y=hourly_fraud.values,
    marker="o"
)

plt.title("Fraud Percentage by Transaction Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Fraud Percentage")

plt.xticks(range(0, 24))

plt.savefig(
    OUTPUT_DIR / "fraud_by_hour.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------
# 8. FINAL SUMMARY
# --------------------------------------------

print("\n========== EDA COMPLETED ==========")

print("Total transactions:", len(train))

print("Legitimate transactions:", (train["is_fraud"] == 0).sum())

print("Fraudulent transactions:", (train["is_fraud"] == 1).sum())

print(
    "Fraud percentage:",
    round(train["is_fraud"].mean() * 100, 4),
    "%"
)

print("\nCharts saved in the outputs folder.")
hour_analysis = (
    train.groupby("transaction_hour")["is_fraud"]
    .agg(
        total_transactions="count",
        fraud_transactions="sum",
        fraud_rate="mean"
    )
)

hour_analysis["fraud_rate"] *= 100

print("\n========== HOURLY FRAUD ANALYSIS ==========")
print(hour_analysis.round(3))