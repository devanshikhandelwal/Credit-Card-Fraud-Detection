
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.sparse import save_npz
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# --------------------------------------------------
# 2. Load trained model and saved preprocessing tools
# --------------------------------------------------

print("Loading trained XGBoost model...")

model = joblib.load(MODEL_DIR / "xgboost_model.pkl")
preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
threshold_settings = joblib.load(MODEL_DIR / "threshold_settings.pkl")

best_f1_threshold = threshold_settings["best_f1_threshold"]
high_precision_threshold = threshold_settings["high_precision_threshold"]

# --------------------------------------------------
# 3. Load untouched test dataset
# --------------------------------------------------

print("\nLoading fraudTest.csv...")

test_df = pd.read_csv(
    OUTPUT_DIR / "test_processed.csv",
    parse_dates=["trans_date_trans_time"]
)

print("Test dataset shape:", test_df.shape)

# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------

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

X_test_raw = test_df[feature_columns]
y_test = test_df["is_fraud"].to_numpy()

# --------------------------------------------------
# 5. Transform test features
# --------------------------------------------------

print("\nTransforming test data...")

X_test = preprocessor.transform(X_test_raw)

print("Transformed test shape:", X_test.shape)

# --------------------------------------------------
# 6. Generate fraud probabilities
# --------------------------------------------------

print("\nGenerating fraud predictions...")

y_probability = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 7. Evaluate selected thresholds
# --------------------------------------------------

thresholds_to_evaluate = {
    "Default (0.5)": 0.5,
    "Best F1": best_f1_threshold,
    "High Precision": high_precision_threshold
}

results = []

print("\n========== FINAL TEST RESULTS ==========")

for strategy, threshold in thresholds_to_evaluate.items():

    y_pred = (y_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        "Strategy": strategy,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    })

    print(f"\n--- {strategy} | Threshold: {threshold:.4f} ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

# --------------------------------------------------
# 8. Calculate threshold-independent metrics
# --------------------------------------------------

roc_auc = roc_auc_score(y_test, y_probability)
pr_auc = average_precision_score(y_test, y_probability)

print("\n========== THRESHOLD-INDEPENDENT METRICS ==========")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")

# --------------------------------------------------
# 9. Save final evaluation results
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "test_evaluation_results.csv",
    index=False
)

summary = pd.DataFrame([{
    "ROC_AUC": roc_auc,
    "PR_AUC": pr_auc
}])

summary.to_csv(
    OUTPUT_DIR / "test_auc_results.csv",
    index=False
)

print("\nResults saved in the outputs folder.")
print("\nFINAL TEST EVALUATION COMPLETED!")
