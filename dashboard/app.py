
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import date, time, datetime

from utils import load_model, predict_transaction


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"



# ==================================================
# 2. CUSTOM CSS - PROFESSIONAL UI
# ==================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background: #0B1120;
    color: #E5E7EB;
}

/* Main content spacing */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* Main headings */
h1 {
    color: #F8FAFC !important;
    font-size: 2.2rem !important;
    font-weight: 750 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #F1F5F9 !important;
    font-weight: 650 !important;
}

/* Paragraphs */
p {
    color: #CBD5E1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #101A2C;
    border-right: 1px solid #25334A;
}

section[data-testid="stSidebar"] h1 {
    color: #F8FAFC !important;
    font-size: 1.7rem !important;
}

/* Sidebar navigation */
section[data-testid="stSidebar"] label {
    color: #CBD5E1 !important;
    font-size: 14px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #17243A,
        #111C2E
    );
    border: 1px solid #293B56;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.15);
    transition: all 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: #38BDF8;
    transform: translateY(-3px);
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 14px !important;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-size: 29px !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(
        90deg,
        #2563EB,
        #0891B2
    );
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 20px;
    font-weight: 650;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(
        90deg,
        #1D4ED8,
        #0E7490
    );
    border: none;
    color: white;
}

/* Input fields */
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] textarea {
    background-color: #17243A !important;
    color: #F8FAFC !important;
    border-color: #334155 !important;
    border-radius: 8px !important;
}

/* Select labels and input labels */
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stDateInput label,
.stTimeInput label {
    color: #CBD5E1 !important;
}

/* Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #293B56 !important;
    border-radius: 13px;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid #293B56;
    border-radius: 10px;
}

/* Horizontal dividers */
hr {
    border-color: #293B56;
}

/* Captions */
.stCaption {
    color: #94A3B8 !important;
}

/* Progress bar */
div[data-testid="stProgressBar"] > div {
    background-color: #1E293B;
}

/* Hide unnecessary Streamlit decoration */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD MODEL AND DATA
# ==================================================

@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def load_analytics_data():
    path = OUTPUT_DIR / "train_processed.csv.gz"

    if not path.exists():
        return None

    return pd.read_csv(path, compression="gzip")


@st.cache_data
def load_model_results():
    path = OUTPUT_DIR / "test_evaluation_results.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


try:
    model, preprocessor, threshold_settings = get_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


analytics_df = load_analytics_data()
results_df = load_model_results()


# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.markdown("# 💳 FraudGuard")
st.sidebar.caption("AI-Powered Fraud Detection System")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Fraud Analytics",
        "Model Performance",
        "Transaction Prediction",
        "CSV Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Model Information")
st.sidebar.success("XGBoost Model Loaded")
st.sidebar.caption("Threshold-based fraud classification")
st.sidebar.caption("Academic project prototype")




# ==================================================
# 5. OVERVIEW PAGE - PROFESSIONAL DESIGN
# ==================================================

if page == "Overview":

    # --------------------------------------------------
    # HERO SECTION
    # --------------------------------------------------

    st.html("""
    <div style="
        background: linear-gradient(120deg, #172B4D, #123B55);
        padding: 32px;
        border-radius: 18px;
        border: 1px solid #28506D;
        margin-bottom: 28px;
    ">
        <div style="
            color: #38BDF8;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 2px;
            margin-bottom: 12px;
        ">
            MACHINE LEARNING • FINANCIAL SECURITY
        </div>

        <h1 style="
            color: #FFFFFF;
            font-size: 34px;
            margin: 12px 0;
        ">
            Credit Card Fraud Detection
        </h1>

        <p style="
            color: #CBD5E1;
            font-size: 16px;
            margin-bottom: 20px;
        ">
            An intelligent XGBoost-powered platform for identifying
            suspicious credit card transactions through machine learning.
        </p>

        <span style="
            background: #164E63;
            color: #67E8F9;
            padding: 8px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        ">
            ● MODEL ACTIVE
        </span>
    </div>
    """)

    # --------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------

    if analytics_df is None:
        st.warning("Training analytics dataset not found.")
        st.stop()

    # --------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------

    total_transactions = len(analytics_df)

    total_fraud = int(
        analytics_df["is_fraud"].sum()
    )

    legitimate = total_transactions - total_fraud

    fraud_rate = (
        total_fraud / total_transactions
    ) * 100

    st.markdown("### Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Fraud Transactions",
        f"{total_fraud:,}"
    )

    col3.metric(
        "Legitimate Transactions",
        f"{legitimate:,}"
    )

    col4.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    st.divider()

    # --------------------------------------------------
    # DISTRIBUTION AND MODEL PERFORMANCE
    # --------------------------------------------------

    left, right = st.columns([1.15, 0.85])

    # Transaction Distribution
    with left:

        st.markdown("### Transaction Distribution")

        fig, ax = plt.subplots(figsize=(7, 4.5))

        fig.patch.set_facecolor("#111C2E")
        ax.set_facecolor("#111C2E")

        ax.pie(
            [legitimate, total_fraud],
            labels=["Legitimate", "Fraud"],
            autopct="%1.2f%%",
            startangle=90,
            colors=["#2563EB", "#F97316"],
            pctdistance=0.78,
            wedgeprops={
                "width": 0.38,
                "edgecolor": "#111C2E",
                "linewidth": 3
            },
            textprops={
                "color": "#E2E8F0",
                "fontsize": 10
            }
        )

        ax.text(
            0,
            0,
            f"{total_transactions:,}\nTransactions",
            ha="center",
            va="center",
            color="white",
            fontsize=13,
            fontweight="bold"
        )

        ax.set_title(
            "Transaction Breakdown",
            color="white",
            fontsize=14,
            pad=18
        )

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    # Model Performance
    with right:

        st.markdown("### Model Performance")

        if results_df is not None:

            # Correct strategy name from test_evaluation_results.csv
            selected = results_df[
                results_df["Strategy"] == "Best F1"
            ]

            if not selected.empty:

                best_result = selected.iloc[0]

                st.html("""
                <div style="
                    background: #17243A;
                    border: 1px solid #293B56;
                    padding: 20px;
                    border-radius: 13px;
                    margin-bottom: 18px;
                ">
                    <div style="
                        color: #94A3B8;
                        font-size: 12px;
                        letter-spacing: 1px;
                    ">
                        SELECTED MACHINE LEARNING MODEL
                    </div>

                    <div style="
                        color: #38BDF8;
                        font-size: 27px;
                        font-weight: 750;
                        margin: 10px 0;
                    ">
                        XGBoost
                    </div>

                    <div style="
                        color: #CBD5E1;
                        font-size: 13px;
                    ">
                        Validation-selected classification threshold
                    </div>
                </div>
                """)

                m1, m2 = st.columns(2)

                m1.metric(
                    "Test Precision",
                    f"{best_result['Precision'] * 100:.2f}%"
                )

                m2.metric(
                    "Test Recall",
                    f"{best_result['Recall'] * 100:.2f}%"
                )

                m3, m4 = st.columns(2)

                m3.metric(
                    "Test F1 Score",
                    f"{best_result['F1_Score']:.4f}"
                )

                m4.metric(
                    "Selected Threshold",
                    f"{best_result['Threshold']:.4f}"
                )

                st.caption(
                    "Performance calculated on the separate test dataset."
                )

            else:
                st.warning(
                    "Best F1 strategy not found in test results."
                )

        else:
            st.warning(
                "Model evaluation results are unavailable."
            )

    st.divider()

    # --------------------------------------------------
    # ABOUT FRAUDGUARD
    # --------------------------------------------------

    st.markdown("### About FraudGuard")

    info1, info2, info3 = st.columns(3)

    with info1:

        st.markdown("#### 🧠 Machine Learning")

        st.write(
            "Uses XGBoost to learn transaction patterns "
            "and identify potentially fraudulent activity."
        )

    with info2:

        st.markdown("#### 📊 Fraud Analytics")

        st.write(
            "Analyzes transaction categories, fraud distribution "
            "and time-based transaction patterns."
        )

    with info3:

        st.markdown("#### ⚡ Transaction Prediction")

        st.write(
            "Provides individual transaction predictions "
            "and bulk CSV-based fraud classification."
        )

    st.divider()

    st.caption(
        "FraudGuard | Credit Card Fraud Detection | "
        "Built with Python, Scikit-learn, XGBoost and Streamlit."
    )

# ==================================================
# END OF OVERVIEW PAGE
# ==================================================



# ==================================================
# 6. FRAUD ANALYTICS - PROFESSIONAL DESIGN
# ==================================================

elif page == "Fraud Analytics":

    st.title("📊 Fraud Analytics")

    st.caption(
        "Explore transaction patterns, fraud categories "
        "and time-based fraud behavior."
    )

    if analytics_df is None:
        st.error("Analytics dataset not found.")
        st.stop()

    # --------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------

    fraud_df = analytics_df[
        analytics_df["is_fraud"] == 1
    ].copy()

    legitimate_df = analytics_df[
        analytics_df["is_fraud"] == 0
    ].copy()

    total_transactions = len(analytics_df)
    total_fraud = len(fraud_df)
    total_legitimate = len(legitimate_df)

    fraud_rate = (
        total_fraud / total_transactions * 100
        if total_transactions else 0
    )

    avg_fraud_amount = (
        fraud_df["amt"].mean() if total_fraud else 0
    )

    avg_legitimate_amount = (
        legitimate_df["amt"].mean()
        if total_legitimate else 0
    )

    # --------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------

    st.markdown("### Fraud Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Fraud Transactions",
        f"{total_fraud:,}"
    )

    col3.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    col4.metric(
        "Average Fraud Amount",
        f"${avg_fraud_amount:,.2f}"
    )

    st.divider()

    # --------------------------------------------------
    # CHART THEME HELPER
    # --------------------------------------------------

    def style_dark_chart(fig, ax, title):

        fig.patch.set_facecolor("#111C2E")
        ax.set_facecolor("#111C2E")

        ax.set_title(
            title,
            color="#F8FAFC",
            fontsize=14,
            fontweight="bold",
            pad=16
        )

        ax.tick_params(
            colors="#CBD5E1",
            labelsize=9
        )

        ax.xaxis.label.set_color("#CBD5E1")
        ax.yaxis.label.set_color("#CBD5E1")

        for spine in ax.spines.values():
            spine.set_color("#334155")

        ax.grid(
            axis="y",
            color="#334155",
            alpha=0.45,
            linestyle="--"
        )

    # --------------------------------------------------
    # FRAUD BY CATEGORY
    # --------------------------------------------------

    st.markdown("### Category and Time-Based Analysis")

    left, right = st.columns(2)

    with left:

        st.markdown("#### Fraud by Transaction Category")

        category_data = (
            analytics_df.groupby("category")["is_fraud"]
            .agg(["mean", "sum", "count"])
            .sort_values("mean", ascending=True)
        )

        category_data["fraud_rate"] = (
            category_data["mean"] * 100
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        fig.patch.set_facecolor("#111C2E")
        ax.set_facecolor("#111C2E")

        bars = ax.barh(
            category_data.index,
            category_data["fraud_rate"],
            color="#38BDF8",
            edgecolor="#0F172A",
            height=0.65
        )

        ax.set_xlabel("Fraud Rate (%)")

        ax.set_xlim(
            0,
            category_data["fraud_rate"].max() * 1.2
        )

        style_dark_chart(
            fig,
            ax,
            "Fraud Rate by Category"
        )

        for bar, value in zip(
            bars,
            category_data["fraud_rate"]
        ):

            ax.text(
                value + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%",
                color="#E2E8F0",
                va="center",
                fontsize=8
            )

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Fraud rate = fraudulent transactions / "
            "total transactions in that category."
        )

    # --------------------------------------------------
    # FRAUD BY HOUR
    # --------------------------------------------------

    with right:

        st.markdown("#### Fraud by Transaction Hour")

        hourly_data = (
            analytics_df.groupby("transaction_hour")["is_fraud"]
            .agg(["mean", "sum", "count"])
            .reindex(range(24), fill_value=0)
        )

        hourly_data["fraud_rate"] = (
            hourly_data["mean"] * 100
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        fig.patch.set_facecolor("#111C2E")
        ax.set_facecolor("#111C2E")

        ax.plot(
            hourly_data.index,
            hourly_data["fraud_rate"],
            color="#2DD4BF",
            linewidth=2.5,
            marker="o",
            markersize=5
        )

        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Fraud Rate (%)")

        ax.set_xticks(range(0, 24, 2))

        style_dark_chart(
            fig,
            ax,
            "Fraud Rate by Hour"
        )

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Hours use the dataset's recorded transaction time. "
            "Fraud rates should be interpreted alongside sample counts."
        )

    # --------------------------------------------------
    # HOURLY TRANSACTION COUNTS
    # --------------------------------------------------

    st.markdown("#### Transaction Volume by Hour")

    fig, ax = plt.subplots(figsize=(12, 3.5))

    fig.patch.set_facecolor("#111C2E")
    ax.set_facecolor("#111C2E")

    ax.bar(
        hourly_data.index,
        hourly_data["count"],
        color="#818CF8",
        width=0.7
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Transaction Count")
    ax.set_xticks(range(24))

    style_dark_chart(
        fig,
        ax,
        "Total Transactions Recorded Each Hour"
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    # --------------------------------------------------
    # TRANSACTION AMOUNT DISTRIBUTION
    # --------------------------------------------------

    st.markdown("### Transaction Amount Analysis")

    left, right = st.columns([1.2, 0.8])

    with left:

        st.markdown("#### Amount Distribution")

        fig, ax = plt.subplots(figsize=(9, 5))

        fig.patch.set_facecolor("#111C2E")
        ax.set_facecolor("#111C2E")

        legitimate_sample = legitimate_df["amt"].sample(
            min(20000, len(legitimate_df)),
            random_state=42
        )

        fraud_sample = fraud_df["amt"].sample(
            min(20000, len(fraud_df)),
            random_state=42
        )

        ax.hist(
            legitimate_sample,
            bins=50,
            alpha=0.7,
            color="#3B82F6",
            label="Legitimate",
            log=True
        )

        ax.hist(
            fraud_sample,
            bins=50,
            alpha=0.7,
            color="#F97316",
            label="Fraud",
            log=True
        )

        ax.set_xlabel("Transaction Amount ($)")
        ax.set_ylabel("Frequency (log scale)")

        style_dark_chart(
            fig,
            ax,
            "Transaction Amount Distribution"
        )

        legend = ax.legend(
            facecolor="#17243A",
            edgecolor="#334155"
        )

        for text in legend.get_texts():
            text.set_color("#F8FAFC")

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # --------------------------------------------------
    # AMOUNT STATISTICS
    # --------------------------------------------------

    with right:

        st.markdown("#### Transaction Statistics")

        st.metric(
            "Average Fraud Amount",
            f"${avg_fraud_amount:,.2f}"
        )

        st.metric(
            "Average Legitimate Amount",
            f"${avg_legitimate_amount:,.2f}"
        )

        st.metric(
            "Highest Fraud Amount",
            f"${fraud_df['amt'].max():,.2f}"
            if total_fraud else "$0.00"
        )

        st.metric(
            "Median Fraud Amount",
            f"${fraud_df['amt'].median():,.2f}"
            if total_fraud else "$0.00"
        )

    st.divider()

    # --------------------------------------------------
    # TOP FRAUD CATEGORIES TABLE
    # --------------------------------------------------

    st.markdown("### Category Risk Summary")

    category_summary = category_data[
        ["count", "sum", "fraud_rate"]
    ].sort_values(
        "fraud_rate",
        ascending=False
    ).reset_index()

    category_summary.columns = [
        "Transaction Category",
        "Total Transactions",
        "Fraud Transactions",
        "Fraud Rate (%)"
    ]

    category_summary["Fraud Rate (%)"] = (
        category_summary["Fraud Rate (%)"].round(3)
    )

    st.dataframe(
        category_summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "All analytics in this section are calculated from "
        "the training dataset and are descriptive, not causal."
    )

# ==================================================
# END OF FRAUD ANALYTICS
# ==================================================



# ==================================================
# 7. MODEL PERFORMANCE - PROFESSIONAL DESIGN
# ==================================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")

    st.caption(
        "Comprehensive evaluation of the XGBoost model "
        "on the separate, untouched test dataset."
    )

    if results_df is None or results_df.empty:
        st.error("Test evaluation results not found.")
        st.stop()

    # --------------------------------------------------
    # LOAD ROC-AUC AND PR-AUC
    # --------------------------------------------------

    auc_path = OUTPUT_DIR / "test_auc_results.csv"

    roc_auc = None
    pr_auc = None

    if auc_path.exists():
        auc_df = pd.read_csv(auc_path)

        if not auc_df.empty:
            roc_auc = float(auc_df.iloc[0]["ROC_AUC"])
            pr_auc = float(auc_df.iloc[0]["PR_AUC"])

    # --------------------------------------------------
    # SELECT CLASSIFICATION STRATEGY
    # --------------------------------------------------

    st.markdown("### Classification Strategy")

    strategy_options = results_df["Strategy"].tolist()

    selected_strategy = st.selectbox(
        "Choose a threshold strategy",
        strategy_options,
        index=1 if "Best F1" in strategy_options else 0
    )

    selected_row = results_df[
        results_df["Strategy"] == selected_strategy
    ].iloc[0]

    threshold = float(selected_row["Threshold"])

    # --------------------------------------------------
    # PRIMARY METRICS
    # --------------------------------------------------

    st.markdown("### Test Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{selected_row['Accuracy'] * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{selected_row['Precision'] * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{selected_row['Recall'] * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{selected_row['F1_Score']:.4f}"
    )

    st.caption(
        f"Selected threshold: {threshold:.4f} | "
        f"Strategy: {selected_strategy}"
    )

    st.divider()

    # --------------------------------------------------
    # ROC-AUC AND PR-AUC
    # --------------------------------------------------

    st.markdown("### Classification Quality")

    auc_col1, auc_col2, auc_col3 = st.columns(3)

    with auc_col1:
        if roc_auc is not None:
            st.metric(
                "ROC-AUC",
                f"{roc_auc:.4f}"
            )
        else:
            st.metric("ROC-AUC", "Unavailable")

    with auc_col2:
        if pr_auc is not None:
            st.metric(
                "PR-AUC",
                f"{pr_auc:.4f}"
            )
        else:
            st.metric("PR-AUC", "Unavailable")

    with auc_col3:
        st.metric(
            "Classification Threshold",
            f"{threshold:.4f}"
        )

    st.caption(
        "ROC-AUC and PR-AUC are threshold-independent metrics "
        "calculated using the model's test-set probabilities. "
        "They remain the same across the selected strategies."
    )

    st.divider()

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    st.markdown("### Confusion Matrix")

    tn = int(selected_row["True_Negatives"])
    fp = int(selected_row["False_Positives"])
    fn = int(selected_row["False_Negatives"])
    tp = int(selected_row["True_Positives"])

    matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    fig, ax = plt.subplots(figsize=(7, 5))

    fig.patch.set_facecolor("#111C2E")
    ax.set_facecolor("#111C2E")

    image = ax.imshow(
        matrix,
        cmap="Blues"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Legitimate", "Fraud"],
        color="#E2E8F0"
    )

    ax.set_yticklabels(
        ["Legitimate", "Fraud"],
        color="#E2E8F0"
    )

    ax.set_xlabel(
        "Predicted Label",
        color="#CBD5E1",
        fontsize=11
    )

    ax.set_ylabel(
        "Actual Label",
        color="#CBD5E1",
        fontsize=11
    )

    ax.set_title(
        "Test Dataset Confusion Matrix",
        color="#F8FAFC",
        fontsize=15,
        fontweight="bold",
        pad=18
    )

    threshold_color = matrix.max() * 0.55

    for i in range(2):
        for j in range(2):

            color = (
                "white"
                if matrix[i, j] > threshold_color
                else "#0F172A"
            )

            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=15,
                fontweight="bold"
            )

    ax.tick_params(colors="#CBD5E1")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    # --------------------------------------------------
    # ERROR ANALYSIS
    # --------------------------------------------------

    st.markdown("### Prediction Error Analysis")

    error_col1, error_col2 = st.columns(2)

    with error_col1:

        st.metric(
            "False Positives",
            f"{fp:,}",
            help="Legitimate transactions incorrectly flagged as fraud."
        )

    with error_col2:

        st.metric(
            "False Negatives",
            f"{fn:,}",
            help="Fraudulent transactions incorrectly classified as legitimate."
        )

    st.divider()

    # --------------------------------------------------
    # STRATEGY COMPARISON
    # --------------------------------------------------

    st.markdown("### Threshold Strategy Comparison")

    comparison = results_df.copy()

    comparison["Precision (%)"] = (
        comparison["Precision"] * 100
    )

    comparison["Recall (%)"] = (
        comparison["Recall"] * 100
    )

    comparison["F1 Score (%)"] = (
        comparison["F1_Score"] * 100
    )

    comparison["Threshold"] = (
        comparison["Threshold"].round(4)
    )

    comparison["Precision (%)"] = (
        comparison["Precision (%)"].round(2)
    )

    comparison["Recall (%)"] = (
        comparison["Recall (%)"].round(2)
    )

    comparison["F1 Score (%)"] = (
        comparison["F1 Score (%)"].round(2)
    )

    st.dataframe(
        comparison[
            [
                "Strategy",
                "Threshold",
                "Precision (%)",
                "Recall (%)",
                "F1 Score (%)",
                "False_Positives",
                "False_Negatives"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # VISUAL METRIC COMPARISON
    # --------------------------------------------------

    st.markdown("### Performance Across Strategies")

    fig, ax = plt.subplots(figsize=(10, 5))

    fig.patch.set_facecolor("#111C2E")
    ax.set_facecolor("#111C2E")

    x = np.arange(len(comparison))
    width = 0.23

    ax.bar(
        x - width,
        comparison["Precision"],
        width,
        label="Precision",
        color="#38BDF8"
    )

    ax.bar(
        x,
        comparison["Recall"],
        width,
        label="Recall",
        color="#2DD4BF"
    )

    ax.bar(
        x + width,
        comparison["F1_Score"],
        width,
        label="F1 Score",
        color="#A78BFA"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        comparison["Strategy"],
        color="#CBD5E1"
    )

    ax.set_ylabel(
        "Score",
        color="#CBD5E1"
    )

    ax.set_ylim(0, 1.08)

    ax.set_title(
        "Precision, Recall and F1 Comparison",
        color="#F8FAFC",
        fontsize=14,
        fontweight="bold"
    )

    ax.tick_params(colors="#CBD5E1")

    ax.grid(
        axis="y",
        color="#334155",
        alpha=0.45,
        linestyle="--"
    )

    legend = ax.legend(
        facecolor="#17243A",
        edgecolor="#334155"
    )

    for text in legend.get_texts():
        text.set_color("#F8FAFC")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    # --------------------------------------------------
    # MODEL INTERPRETATION
    # --------------------------------------------------

    st.markdown("### Model Evaluation Insights")

    if selected_strategy == "Best F1":

        st.info(
            "The selected strategy balances precision and recall. "
            "On the test dataset, the model achieved an F1 score "
            "of 0.7849 at the validation-selected threshold."
        )

    elif selected_strategy == "High Precision":

        st.info(
            "This strategy prioritizes detecting more fraud than "
            "the best-F1 strategy, at the cost of additional false "
            "positive alerts."
        )

    else:

        st.info(
            "The default threshold prioritizes fraud recall, "
            "but generates more false positive alerts."
        )

    st.caption(
        "Evaluation is based on the provided historical dataset. "
        "Results represent an academic model prototype and do not "
        "establish real-world production performance."
    )

# ==================================================
# END OF MODEL PERFORMANCE
# ==================================================



# ==================================================
# 8. TRANSACTION PREDICTION - PROFESSIONAL DESIGN
# ==================================================

elif page == "Transaction Prediction":

    st.title("🔍 Transaction Risk Analysis")

    st.caption(
        "Analyze an individual transaction using the trained "
        "XGBoost fraud detection model."
    )

    # --------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------

    st.html("""
    <div style="
        background: linear-gradient(120deg, #172B4D, #123B55);
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #28506D;
        margin-bottom: 22px;
    ">
        <div style="
            color: #38BDF8;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1.5px;
        ">
            INTELLIGENT TRANSACTION SCREENING
        </div>

        <h3 style="color:#FFFFFF;margin:10px 0;">
            Evaluate Transaction Risk
        </h3>

        <p style="color:#CBD5E1;margin-bottom:0;">
            Enter transaction, customer and location details.
            FraudGuard will process the features and generate
            a prediction using the trained XGBoost model.
        </p>
    </div>
    """)

    st.markdown("### Transaction Information")

    # --------------------------------------------------
    # TRANSACTION FORM
    # --------------------------------------------------

    with st.form("transaction_prediction_form"):

        # Transaction details
        st.markdown("#### 01 — Transaction Details")

        col1, col2 = st.columns(2)

        with col1:

            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01,
                value=100.0,
                step=10.0,
                help="Enter the transaction amount."
            )

            category_options = [
                "grocery_pos",
                "shopping_pos",
                "misc_pos",
                "shopping_net",
                "misc_net",
                "grocery_net",
                "food_dining",
                "entertainment",
                "gas_transport",
                "personal_care",
                "health_fitness",
                "home",
                "kids_pets",
                "travel"
            ]

            category = st.selectbox(
                "Transaction Category",
                category_options
            )

            transaction_date = st.date_input(
                "Transaction Date",
                value=date(2020, 6, 1)
            )

        with col2:

            transaction_time = st.time_input(
                "Transaction Time",
                value=time(22, 30)
            )

            gender = st.selectbox(
                "Customer Gender",
                ["F", "M"]
            )

            city_population = st.number_input(
                "City Population",
                min_value=0,
                value=50000,
                step=1000
            )

        st.divider()

        # Customer information
        st.markdown("#### 02 — Customer Information")

        col3, col4 = st.columns(2)

        with col3:

            state = st.text_input(
                "State Code",
                value="NY",
                max_chars=2,
                help="Enter a two-letter state code."
            ).strip().upper()

            job = st.text_input(
                "Customer Occupation",
                value="Engineer"
            ).strip()

        with col4:

            dob = st.date_input(
                "Customer Date of Birth",
                value=date(1985, 1, 1)
            )

        st.divider()

        # Location information
        st.markdown("#### 03 — Transaction Location")

        st.caption(
            "Enter the customer and merchant coordinates "
            "to calculate the distance between them."
        )

        customer_col, merchant_col = st.columns(2)

        with customer_col:

            st.markdown("**Customer Location**")

            lat = st.number_input(
                "Customer Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=40.7128,
                format="%.6f"
            )

            longitude = st.number_input(
                "Customer Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-74.0060,
                format="%.6f"
            )

        with merchant_col:

            st.markdown("**Merchant Location**")

            merch_lat = st.number_input(
                "Merchant Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=40.7306,
                format="%.6f"
            )

            merch_long = st.number_input(
                "Merchant Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-73.9352,
                format="%.6f"
            )

        st.divider()

        # Threshold selection
        st.markdown("#### 04 — Classification Strategy")

        threshold_options = {
            "Balanced — Best F1": (
                threshold_settings["best_f1_threshold"]
            ),
            "High Recall — High Precision Strategy": (
                threshold_settings["high_precision_threshold"]
            ),
            "Default — 0.50": 0.5
        }

        threshold_choice = st.selectbox(
            "Select Detection Strategy",
            list(threshold_options.keys()),
            index=0
        )

        selected_threshold = threshold_options[
            threshold_choice
        ]

        st.caption(
            f"Selected classification threshold: "
            f"{selected_threshold:.4f}"
        )

        submitted = st.form_submit_button(
            "Analyze Transaction",
            use_container_width=True
        )

    # --------------------------------------------------
    # PREDICTION PROCESSING
    # --------------------------------------------------

    if submitted:

        if not state or len(state) != 2:
            st.error(
                "Please enter a valid two-letter state code."
            )

        elif not job:
            st.error(
                "Please enter the customer's occupation."
            )

        elif dob >= transaction_date:
            st.error(
                "Customer date of birth must be before "
                "the transaction date."
            )

        else:

            transaction = {
                "amt": amount,
                "category": category,
                "gender": gender,
                "state": state,
                "job": job,
                "city_pop": city_population,

                "trans_date_trans_time": datetime.combine(
                    transaction_date,
                    transaction_time
                ),

                "dob": dob,

                "lat": lat,
                "long": longitude,
                "merch_lat": merch_lat,
                "merch_long": merch_long
            }

            try:

                with st.spinner(
                    "Analyzing transaction with XGBoost..."
                ):

                    result = predict_transaction(
                        transaction,
                        model,
                        preprocessor,
                        selected_threshold
                    )

                score = result["fraud_probability"]
                is_fraud = result["prediction"]

                st.divider()

                st.markdown("### Prediction Results")

                # --------------------------------------------------
                # RESULT HEADER
                # --------------------------------------------------

                if is_fraud == 1:

                    st.error(
                        "⚠️ Potentially Fraudulent Transaction"
                    )

                    status_text = "Potential Fraud"

                else:

                    st.success(
                        "✅ Classified as Legitimate"
                    )

                    status_text = "Legitimate"

                # --------------------------------------------------
                # RESULT METRICS
                # --------------------------------------------------

                result_col1, result_col2, result_col3 = (
                    st.columns(3)
                )

                result_col1.metric(
                    "Model Fraud Score",
                    f"{score * 100:.2f}%"
                )

                result_col2.metric(
                    "Classification Threshold",
                    f"{selected_threshold:.4f}"
                )

                result_col3.metric(
                    "Predicted Status",
                    status_text
                )

                # --------------------------------------------------
                # SCORE VISUALIZATION
                # --------------------------------------------------

                st.markdown("#### Fraud Score")

                st.progress(
                    min(max(float(score), 0.0), 1.0)
                )

                st.caption(
                    "The score is the model's predicted fraud probability. "
                    "It is not a guarantee that the transaction is fraudulent."
                )

                # --------------------------------------------------
                # DECISION EXPLANATION
                # --------------------------------------------------

                if is_fraud == 1:

                    st.html("""
                    <div style="
                        background:#451A1A;
                        border:1px solid #991B1B;
                        padding:18px;
                        border-radius:12px;
                    ">
                        <h4 style="color:#FCA5A5;margin-top:0;">
                            Transaction Flagged
                        </h4>
                        <p style="color:#FECACA;margin-bottom:0;">
                            The model score met or exceeded the selected
                            classification threshold. This transaction
                            should be reviewed further.
                        </p>
                    </div>
                    """)

                else:

                    st.html("""
                    <div style="
                        background:#12352B;
                        border:1px solid #166534;
                        padding:18px;
                        border-radius:12px;
                    ">
                        <h4 style="color:#86EFAC;margin-top:0;">
                            Transaction Classified as Legitimate
                        </h4>
                        <p style="color:#BBF7D0;margin-bottom:0;">
                            The model score is below the selected
                            classification threshold.
                        </p>
                    </div>
                    """)

                # --------------------------------------------------
                # TRANSACTION SUMMARY
                # --------------------------------------------------

                with st.expander(
                    "View Analyzed Transaction Details"
                ):

                    summary = pd.DataFrame({
                        "Feature": [
                            "Transaction Amount",
                            "Category",
                            "Transaction Date",
                            "Transaction Time",
                            "Customer Gender",
                            "State",
                            "Occupation",
                            "City Population",
                            "Customer Coordinates",
                            "Merchant Coordinates"
                        ],

                        "Value": [
                            f"${amount:,.2f}",
                            category,
                            str(transaction_date),
                            str(transaction_time),
                            gender,
                            state,
                            job,
                            f"{city_population:,}",
                            f"{lat}, {longitude}",
                            f"{merch_lat}, {merch_long}"
                        ]
                    })

                    st.dataframe(
                        summary,
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as error:

                st.error(
                    f"Transaction prediction failed: {error}"
                )

    # --------------------------------------------------
    # FOOTNOTE
    # --------------------------------------------------

    st.divider()

    st.caption(
        "FraudGuard uses historical transaction data and a trained "
        "XGBoost classifier. This is an academic demonstration, "
        "not a live banking authorization or fraud investigation system."
    )

# ==================================================
# END OF TRANSACTION PREDICTION
# ==================================================



# ==================================================
# 9. CSV BULK PREDICTION - PROFESSIONAL DESIGN
# ==================================================

elif page == "CSV Prediction":

    st.title("📁 Bulk Fraud Detection")

    st.caption(
        "Upload transaction records and analyze multiple "
        "transactions using the trained XGBoost model."
    )

    # --------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------

    st.html("""
    <div style="
        background: linear-gradient(120deg, #172B4D, #123B55);
        padding: 24px;
        border-radius: 15px;
        border: 1px solid #28506D;
        margin-bottom: 22px;
    ">
        <div style="
            color: #38BDF8;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1.5px;
        ">
            BATCH TRANSACTION SCREENING
        </div>

        <h3 style="color:#FFFFFF;margin:10px 0;">
            Analyze Multiple Transactions
        </h3>

        <p style="color:#CBD5E1;margin-bottom:0;">
            Upload a transaction CSV file, validate its contents,
            generate fraud probabilities and download the complete
            prediction report.
        </p>
    </div>
    """)

    # --------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------

    required_columns = [
        "amt",
        "category",
        "gender",
        "state",
        "job",
        "city_pop",
        "trans_date_trans_time",
        "dob",
        "lat",
        "long",
        "merch_lat",
        "merch_long"
    ]

    # --------------------------------------------------
    # THRESHOLD SELECTION
    # --------------------------------------------------

    st.markdown("### Classification Strategy")

    csv_threshold_options = {
        "Balanced — Best F1": (
            threshold_settings["best_f1_threshold"]
        ),
        "High Recall — High Precision Strategy": (
            threshold_settings["high_precision_threshold"]
        ),
        "Default — 0.50": 0.5
    }

    csv_strategy = st.selectbox(
        "Select prediction threshold",
        list(csv_threshold_options.keys()),
        index=0,
        key="csv_threshold_strategy"
    )

    csv_threshold = csv_threshold_options[csv_strategy]

    st.caption(
        f"Selected threshold: {csv_threshold:.4f}"
    )

    st.divider()

    # --------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------

    st.markdown("### Upload Transaction Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV with the raw transaction columns required by FraudGuard."
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV file to begin bulk fraud detection."
        )

        with st.expander("View required CSV columns"):

            st.write(
                "Your uploaded file must contain these columns:"
            )

            st.code(
                "\n".join(required_columns),
                language="text"
            )

        st.caption(
            "The original fraudTrain.csv and fraudTest.csv "
            "files contain the required raw transaction fields."
        )

    else:

        # --------------------------------------------------
        # READ CSV
        # --------------------------------------------------

        try:

            upload_df = pd.read_csv(uploaded_file)

            st.success(
                f"File uploaded successfully: {uploaded_file.name}"
            )

            st.markdown("### Dataset Preview")

            preview_col1, preview_col2, preview_col3 = (
                st.columns(3)
            )

            preview_col1.metric(
                "Total Rows",
                f"{len(upload_df):,}"
            )

            preview_col2.metric(
                "Total Columns",
                f"{len(upload_df.columns):,}"
            )

            preview_col3.metric(
                "Missing Values",
                f"{int(upload_df.isnull().sum().sum()):,}"
            )

            st.dataframe(
                upload_df.head(10),
                use_container_width=True,
                hide_index=True
            )

            # --------------------------------------------------
            # VALIDATE DATASET
            # --------------------------------------------------

            st.markdown("### Dataset Validation")

            missing_columns = [
                col for col in required_columns
                if col not in upload_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

                st.stop()

            if len(upload_df) == 0:

                st.error("The uploaded CSV file is empty.")
                st.stop()

            if len(upload_df) > 5000:

                st.warning(
                    "Maximum upload limit is 5,000 transactions "
                    "per batch. Please upload a smaller CSV."
                )

                st.stop()

            if upload_df[required_columns].isnull().any().any():

                missing_summary = (
                    upload_df[required_columns]
                    .isnull()
                    .sum()
                )

                missing_summary = missing_summary[
                    missing_summary > 0
                ]

                st.error(
                    "Missing values were found in required columns."
                )

                st.dataframe(
                    missing_summary.rename(
                        "Missing Values"
                    ),
                    use_container_width=True
                )

                st.stop()

            # Validate numeric columns
            numeric_columns = [
                "amt",
                "city_pop",
                "lat",
                "long",
                "merch_lat",
                "merch_long"
            ]

            numeric_data = upload_df[
                numeric_columns
            ].apply(
                pd.to_numeric,
                errors="coerce"
            )

            if numeric_data.isnull().any().any():

                st.error(
                    "One or more numeric fields contain invalid values."
                )

                st.stop()

            if (
                (numeric_data["amt"] <= 0).any()
                or (numeric_data["city_pop"] < 0).any()
            ):

                st.error(
                    "Transaction amount must be positive and "
                    "city population cannot be negative."
                )

                st.stop()

            if (
                (~numeric_data["lat"].between(-90, 90)).any()
                or (~numeric_data["merch_lat"].between(-90, 90)).any()
                or (~numeric_data["long"].between(-180, 180)).any()
                or (~numeric_data["merch_long"].between(-180, 180)).any()
            ):

                st.error(
                    "Invalid latitude or longitude values found."
                )

                st.stop()

            # Validate date fields
            transaction_dates = pd.to_datetime(
                upload_df["trans_date_trans_time"],
                errors="coerce"
            )

            birth_dates = pd.to_datetime(
                upload_df["dob"],
                errors="coerce"
            )

            if (
                transaction_dates.isnull().any()
                or birth_dates.isnull().any()
            ):

                st.error(
                    "Invalid transaction date or date of birth found."
                )

                st.stop()

            if (birth_dates >= transaction_dates).any():

                st.error(
                    "Some customers have a date of birth "
                    "after their transaction date."
                )

                st.stop()

            st.success(
                "Dataset validation passed! "
                "All required fields are available."
            )

            # --------------------------------------------------
            # PREDICTION BUTTON
            # --------------------------------------------------

            st.divider()

            st.markdown("### Generate Predictions")

            st.write(
                f"Ready to analyze {len(upload_df):,} transactions "
                f"using the {csv_strategy} strategy."
            )

            if st.button(
                "🚀 Analyze All Transactions",
                use_container_width=True,
                key="run_bulk_predictions"
            ):

                try:

                    progress_bar = st.progress(0)

                    status_text = st.empty()

                    predictions = []

                    total_rows = len(upload_df)

                    for index, record in enumerate(
                        upload_df.to_dict(orient="records")
                    ):

                        result = predict_transaction(
                            record,
                            model,
                            preprocessor,
                            csv_threshold
                        )

                        predictions.append(result)

                        if (
                            (index + 1) % 25 == 0
                            or index + 1 == total_rows
                        ):

                            progress = (
                                (index + 1) / total_rows
                            )

                            progress_bar.progress(progress)

                            status_text.write(
                                f"Analyzing transaction "
                                f"{index + 1:,} of {total_rows:,}"
                            )

                    # --------------------------------------------------
                    # BUILD RESULT DATAFRAME
                    # --------------------------------------------------

                    result_df = upload_df.copy()

                    result_df["fraud_probability"] = [
                        item["fraud_probability"]
                        for item in predictions
                    ]

                    result_df["prediction"] = [
                        item["prediction"]
                        for item in predictions
                    ]

                    result_df["status"] = [
                        item["status"]
                        for item in predictions
                    ]

                    progress_bar.empty()
                    status_text.empty()

                    # --------------------------------------------------
                    # STORE RESULTS FOR SESSION
                    # --------------------------------------------------

                    st.session_state["fraudguard_csv_results"] = (
                        result_df
                    )

                    st.session_state["fraudguard_csv_strategy"] = (
                        csv_strategy
                    )

                    st.success(
                        "Bulk fraud analysis completed successfully!"
                    )

                
                
                except Exception as error:

                    st.error(
                        f"Bulk prediction failed: {error}"
                    )

        except Exception as error:

            st.error(
                f"Could not read or validate the uploaded CSV: {error}"
            )

    # --------------------------------------------------
    # DISPLAY PREDICTION RESULTS
    # --------------------------------------------------



    if "fraudguard_csv_results" in st.session_state:

        result_df = st.session_state[
            "fraudguard_csv_results"
        ]

        st.divider()

        st.markdown("### Prediction Summary")

        total_analyzed = len(result_df)

        fraud_count = int(
            (result_df["prediction"] == 1).sum()
        )

        legitimate_count = (
            total_analyzed - fraud_count
        )

        predicted_fraud_rate = (
            fraud_count / total_analyzed * 100
            if total_analyzed else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Transactions Analyzed",
            f"{total_analyzed:,}"
        )

        col2.metric(
            "Potential Fraud",
            f"{fraud_count:,}"
        )

        col3.metric(
            "Legitimate",
            f"{legitimate_count:,}"
        )

        col4.metric(
            "Predicted Fraud Rate",
            f"{predicted_fraud_rate:.2f}%"
        )

        # --------------------------------------------------
        # PREDICTION DISTRIBUTION
        # --------------------------------------------------

        left, right = st.columns(2)

        with left:

            st.markdown("#### Prediction Distribution")

            fig, ax = plt.subplots(figsize=(6, 4))

            fig.patch.set_facecolor("#111C2E")
            ax.set_facecolor("#111C2E")

            ax.pie(
                [legitimate_count, fraud_count],
                labels=["Legitimate", "Potential Fraud"],
                autopct="%1.1f%%",
                colors=["#2563EB", "#F97316"],
                startangle=90,
                wedgeprops={
                    "width": 0.38,
                    "edgecolor": "#111C2E",
                    "linewidth": 3
                },
                textprops={
                    "color": "#F8FAFC"
                }
            )

            ax.set_title(
                "Predicted Transaction Classes",
                color="#F8FAFC"
            )

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:

            st.markdown("#### Fraud Probability Distribution")

            fig, ax = plt.subplots(figsize=(6, 4))

            fig.patch.set_facecolor("#111C2E")
            ax.set_facecolor("#111C2E")

            ax.hist(
                result_df["fraud_probability"] * 100,
                bins=25,
                color="#38BDF8",
                edgecolor="#111C2E"
            )

            ax.set_xlabel(
                "Fraud Probability (%)",
                color="#CBD5E1"
            )

            ax.set_ylabel(
                "Number of Transactions",
                color="#CBD5E1"
            )

            ax.set_title(
                "Fraud Probability Distribution",
                color="#F8FAFC"
            )

            ax.tick_params(colors="#CBD5E1")

            ax.grid(
                axis="y",
                alpha=0.25,
                color="#64748B"
            )

            for spine in ax.spines.values():
                spine.set_color("#334155")

            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # --------------------------------------------------
        # RESULTS TABLE
        # --------------------------------------------------

        st.divider()

        st.markdown("### Transaction Prediction Results")

        fraud_filter = st.selectbox(
            "Filter Transactions",
            [
                "All Transactions",
                "Potential Fraud Only",
                "Legitimate Only"
            ],
            key="csv_result_filter"
        )

        if fraud_filter == "Potential Fraud Only":

            display_df = result_df[
                result_df["prediction"] == 1
            ]

        elif fraud_filter == "Legitimate Only":

            display_df = result_df[
                result_df["prediction"] == 0
            ]

        else:

            display_df = result_df

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # DOWNLOAD REPORT
        # --------------------------------------------------

        st.markdown("### Export Prediction Report")

        csv_output = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download Complete Prediction Report",
            data=csv_output,
            file_name="fraudguard_prediction_report.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "The report includes original transaction fields, "
            "fraud probability, predicted class and classification status."
        )

# ==================================================
# END OF CSV BULK PREDICTION
# ==================================================
