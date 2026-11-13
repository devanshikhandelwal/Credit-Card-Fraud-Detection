
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

# Load final test results
df = pd.read_csv(OUTPUT_DIR / "test_evaluation_results.csv")

# Make strategy names shorter for charts
df["Strategy"] = [
    "Default (0.5)",
    "Validation-selected\n(0.9551)",
    "High precision\n(0.8395)"
]

# --------------------------------------------------
# 1. Compare Precision, Recall and F1
# --------------------------------------------------

metrics = ["Precision", "Recall", "F1_Score"]

ax = df.set_index("Strategy")[metrics].plot(
    kind="bar",
    figsize=(11, 6),
    width=0.75
)

ax.set_title("XGBoost: Final Test Performance Comparison")
ax.set_xlabel("Threshold Strategy")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend(["Precision", "Recall", "F1 Score"])
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "test_metrics_comparison.png",
    dpi=300
)
plt.show()

# --------------------------------------------------
# 2. Compare False Positives and False Negatives
# --------------------------------------------------

ax = df.set_index("Strategy")[
    ["False_Positives", "False_Negatives"]
].plot(
    kind="bar",
    figsize=(11, 6),
    width=0.75
)

ax.set_title("False Positives vs False Negatives")
ax.set_xlabel("Threshold Strategy")
ax.set_ylabel("Number of Transactions")
ax.legend(["False Positives", "False Negatives"])
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_comparison.png",
    dpi=300
)
plt.show()

# --------------------------------------------------
# 3. Plot Confusion Matrices
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, row in df.iterrows():

    tn = int(row["True_Negatives"])
    fp = int(row["False_Positives"])
    fn = int(row["False_Negatives"])
    tp = int(row["True_Positives"])

    matrix = [
        [tn, fp],
        [fn, tp]
    ]

    image = axes[i].imshow(matrix, cmap="Blues")

    axes[i].set_title(row["Strategy"])
    axes[i].set_xlabel("Predicted Label")
    axes[i].set_ylabel("Actual Label")

    axes[i].set_xticks([0, 1])
    axes[i].set_yticks([0, 1])
    axes[i].set_xticklabels(["Legitimate", "Fraud"])
    axes[i].set_yticklabels(["Legitimate", "Fraud"])

    for r in range(2):
        for c in range(2):
            axes[i].text(
                c,
                r,
                f"{matrix[r][c]:,}",
                ha="center",
                va="center",
                color="black",
                fontsize=11
            )

fig.suptitle("XGBoost Confusion Matrices - Test Dataset")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "test_confusion_matrices.png",
    dpi=300
)
plt.show()

print("\nAll final evaluation visualizations saved successfully!")
