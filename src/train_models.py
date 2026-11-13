
# ============================================
# CREDIT CARD FRAUD DETECTION
# PHASE 5: LOGISTIC REGRESSION
# ============================================

import numpy as np
import pandas as pd
import joblib
import time

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score
)

# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# ============================================
# 2. LOAD PREPARED DATA
# ============================================

print("Loading training and validation data...")

X_train = load_npz(OUTPUT_DIR / "X_train.npz")
X_val = load_npz(OUTPUT_DIR / "X_val.npz")

y_train = np.load(OUTPUT_DIR / "y_train.npy")
y_val = np.load(OUTPUT_DIR / "y_val.npy")

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# ============================================
# 3. CLASS DISTRIBUTION
# ============================================

print("\nTraining class distribution:")

unique, counts = np.unique(
    y_train,
    return_counts=True
)

for label, count in zip(unique, counts):
    print(f"Class {label}: {count}")

# ============================================
# 4. BUILD LOGISTIC REGRESSION MODEL
# ============================================

print("\nInitializing Logistic Regression...")

model = LogisticRegression(
    class_weight="balanced",
    solver="saga",
    max_iter=500,
    tol=1e-3,
    C=1.0,
    random_state=42
)

# ============================================
# 5. TRAIN MODEL
# ============================================

print("\nTraining model...")

start_time = time.time()

model.fit(X_train, y_train)

end_time = time.time()

training_time = end_time - start_time

print("\nModel training completed!")

print(
    f"Training time: {training_time:.2f} seconds"
)

# ============================================
# 6. PREDICTIONS
# ============================================

print("\nGenerating validation predictions...")

y_pred = model.predict(X_val)

y_probability = model.predict_proba(X_val)[:, 1]

# ============================================
# 7. EVALUATION
# ============================================

accuracy = accuracy_score(y_val, y_pred)

precision = precision_score(
    y_val,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_val,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_val,
    y_probability
)

pr_auc = average_precision_score(
    y_val,
    y_probability
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

# ============================================
# 8. CLASSIFICATION REPORT
# ============================================

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_pred,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)

# ============================================
# 9. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_val, y_pred)

print("\nConfusion Matrix:")

print(cm)

# ============================================
# 10. SAVE MODEL
# ============================================

joblib.dump(
    model,
    MODEL_DIR / "logistic_regression.pkl"
)

print("\nModel saved successfully!")

# ============================================
# 11. SAVE PERFORMANCE RESULTS
# ============================================

results = pd.DataFrame([{
    "Model": "Logistic Regression",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": roc_auc,
    "PR_AUC": pr_auc,
    "Training_Time": training_time
}])

results.to_csv(
    OUTPUT_DIR / "model_results.csv",
    index=False
)

print("\nPerformance results saved!")

print("\nPHASE 5 BASELINE MODEL COMPLETED!")
