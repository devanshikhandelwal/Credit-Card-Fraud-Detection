
# ============================================
# CREDIT CARD FRAUD DETECTION
# PHASE 6: THRESHOLD OPTIMIZATION
# ============================================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.sparse import load_npz

from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# ============================================
# 2. LOAD MODEL AND VALIDATION DATA
# ============================================

print("Loading XGBoost model...")

model = joblib.load(
    MODEL_DIR / "xgboost_model.pkl"
)

X_val = load_npz(
    OUTPUT_DIR / "X_val.npz"
)

y_val = np.load(
    OUTPUT_DIR / "y_val.npy"
)

# ============================================
# 3. PREDICT FRAUD PROBABILITIES
# ============================================

print("\nGenerating fraud probabilities...")

y_probability = model.predict_proba(X_val)[:, 1]

# ============================================
# 4. PRECISION-RECALL CURVE
# ============================================

precision_values, recall_values, thresholds = (
    precision_recall_curve(
        y_val,
        y_probability
    )
)

# Calculate F1 for each threshold
f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    /
    (
        precision_values[:-1]
        + recall_values[:-1]
        + 1e-10
    )
)

best_f1_index = np.argmax(f1_values)

best_f1_threshold = thresholds[best_f1_index]

# ============================================
# 5. THRESHOLD EVALUATION
# ============================================

thresholds_to_test = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    best_f1_threshold
]

results = []

for threshold in thresholds_to_test:

    predictions = (
        y_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    })

results_df = pd.DataFrame(results)

results_df = results_df.drop_duplicates(
    subset=["Threshold"]
)

results_df = results_df.sort_values(
    "Threshold"
)

print("\n========== THRESHOLD COMPARISON ==========")

print(
    results_df.round(4).to_string(index=False)
)

# ============================================
# 6. BEST F1 THRESHOLD
# ============================================

best_predictions = (
    y_probability >= best_f1_threshold
).astype(int)

print("\n========== BEST F1 THRESHOLD ==========")

print("Threshold:", round(best_f1_threshold, 4))

print(
    "Precision:",
    round(precision_score(
        y_val,
        best_predictions,
        zero_division=0
    ), 4)
)

print(
    "Recall:",
    round(recall_score(
        y_val,
        best_predictions,
        zero_division=0
    ), 4)
)

print(
    "F1 Score:",
    round(f1_score(
        y_val,
        best_predictions,
        zero_division=0
    ), 4)
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_val,
        best_predictions
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        best_predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)

# ============================================
# 7. HIGH-PRECISION STRATEGY
# ============================================

# Select the highest recall threshold
# that achieves at least 70% precision.

valid_indices = np.where(
    precision_values[:-1] >= 0.70
)[0]

if len(valid_indices) > 0:

    high_precision_index = valid_indices[
        np.argmax(recall_values[valid_indices])
    ]

    high_precision_threshold = thresholds[
        high_precision_index
    ]

    print("\n========== HIGH PRECISION STRATEGY ==========")

    print(
        "Threshold:",
        round(high_precision_threshold, 4)
    )

    print(
        "Precision:",
        round(precision_values[high_precision_index], 4)
    )

    print(
        "Recall:",
        round(recall_values[high_precision_index], 4)
    )

else:

    high_precision_threshold = None

    print("\nNo threshold achieved 70% precision.")

# ============================================
# 8. SAVE RESULTS
# ============================================

results_df.to_csv(
    OUTPUT_DIR / "threshold_results.csv",
    index=False
)

# Save the best F1 threshold
joblib.dump(
    {
        "best_f1_threshold": float(best_f1_threshold),
        "high_precision_threshold": (
            float(high_precision_threshold)
            if high_precision_threshold is not None
            else None
        )
    },
    MODEL_DIR / "threshold_settings.pkl"
)

# ============================================
# 9. VISUALIZE PRECISION AND RECALL
# ============================================

plt.figure(figsize=(10, 6))

plt.plot(
    thresholds,
    precision_values[:-1],
    label="Precision"
)

plt.plot(
    thresholds,
    recall_values[:-1],
    label="Recall"
)

plt.plot(
    thresholds,
    f1_values,
    label="F1 Score"
)

plt.axvline(
    best_f1_threshold,
    linestyle="--",
    label="Best F1 Threshold"
)

plt.title("Precision, Recall and F1 vs Threshold")

plt.xlabel("Classification Threshold")
plt.ylabel("Score")

plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    OUTPUT_DIR / "threshold_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nThreshold results saved!")
print("\nTHRESHOLD OPTIMIZATION COMPLETED!")
