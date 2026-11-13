
# ============================================
# CREDIT CARD FRAUD DETECTION
# MODEL 3: XGBOOST
# ============================================

import numpy as np
import pandas as pd
import joblib
import time

from pathlib import Path
from scipy.sparse import load_npz

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# ============================================
# 2. LOAD DATA
# ============================================

print("Loading prepared data...")

X_train = load_npz(OUTPUT_DIR / "X_train.npz")
X_val = load_npz(OUTPUT_DIR / "X_val.npz")

y_train = np.load(OUTPUT_DIR / "y_train.npy")
y_val = np.load(OUTPUT_DIR / "y_val.npy")

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# ============================================
# 3. STRATIFIED SAMPLE
# ============================================

print("\nCreating stratified sample...")

X_sample, _, y_sample, _ = train_test_split(
    X_train,
    y_train,
    train_size=250000,
    stratify=y_train,
    random_state=42
)

print("Sample shape:", X_sample.shape)

print("\nClass distribution:")
print(pd.Series(y_sample).value_counts())

# ============================================
# 4. CALCULATE CLASS WEIGHT
# ============================================

negative_count = np.sum(y_sample == 0)
positive_count = np.sum(y_sample == 1)

scale_pos_weight = negative_count / positive_count

print(
    "\nScale positive weight:",
    round(scale_pos_weight, 2)
)

# ============================================
# 5. INITIALIZE XGBOOST
# ============================================

print("\nInitializing XGBoost...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

# ============================================
# 6. TRAIN MODEL
# ============================================

print("\nTraining XGBoost...")

start_time = time.time()

model.fit(X_sample, y_sample)

training_time = time.time() - start_time

print("\nXGBoost training completed!")

print(
    f"Training time: {training_time:.2f} seconds"
)

# ============================================
# 7. PREDICTIONS
# ============================================

print("\nGenerating validation predictions...")

y_pred = model.predict(X_val)

y_probability = model.predict_proba(X_val)[:, 1]

# ============================================
# 8. EVALUATION
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

print("\n========== XGBOOST RESULTS ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

# ============================================
# 9. CLASSIFICATION REPORT
# ============================================

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)

# ============================================
# 10. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_val, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ============================================
# 11. SAVE MODEL
# ============================================

joblib.dump(
    model,
    MODEL_DIR / "xgboost_model.pkl"
)

print("\nXGBoost model saved!")

# ============================================
# 12. SAVE RESULTS
# ============================================

result = pd.DataFrame([{
    "Model": "XGBoost",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": roc_auc,
    "PR_AUC": pr_auc,
    "Training_Time": training_time
}])

results_path = OUTPUT_DIR / "model_results.csv"

if results_path.exists():
    previous = pd.read_csv(results_path)

    previous = previous[
        previous["Model"] != "XGBoost"
    ]

    result = pd.concat(
        [previous, result],
        ignore_index=True
    )

result.to_csv(
    results_path,
    index=False
)

print("\nResults saved successfully!")

print("\nXGBOOST TRAINING COMPLETED!")
