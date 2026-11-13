# 💳 FraudGuard — Credit Card Fraud Detection System

### An End-to-End Machine Learning Application for Detecting Potentially Fraudulent Transactions

<p align="center">

**Python | Machine Learning | XGBoost | Scikit-learn | Streamlit**

</p>

<p align="center">

[🚀 Live Application](https://fraudsguard.streamlit.app/) · [💻 GitHub Repository](https://github.com/devanshikhandelwal/Credit-Card-Fraud-Detection)

</p>

---

## 🌐 Live Demo

**Try FraudGuard:** https://fraudsguard.streamlit.app/

FraudGuard is an interactive machine learning application that analyzes credit card transactions and predicts whether a transaction may be fraudulent.

The application provides transaction analytics, model performance evaluation, individual transaction prediction, and bulk CSV-based fraud detection through an interactive Streamlit dashboard.

## 📌 Project Overview

The increasing use of digital payments has made financial fraud detection an important challenge. Identifying suspicious transactions accurately is difficult because fraudulent transactions represent only a small proportion of overall transaction activity.

FraudGuard addresses this problem using machine learning classification techniques.

The project follows a complete machine learning workflow, beginning with exploratory data analysis and feature engineering, followed by model training, evaluation, threshold optimization, and deployment.

Three classification algorithms were implemented and compared:

- Logistic Regression
- Random Forest
- XGBoost

XGBoost was selected as the final model based on its strong validation performance.

The trained model is integrated into a Streamlit application that allows users to analyze individual transactions and upload CSV files for bulk predictions.

## 🎯 Project Objectives

- Understand transaction patterns through exploratory data analysis.
- Identify patterns associated with fraudulent transactions.
- Perform feature engineering using transaction, temporal, and geographical information.
- Handle severe class imbalance.
- Train and compare multiple machine learning algorithms.
- Optimize classification thresholds using validation data.
- Evaluate the selected model on an independent test dataset.
- Develop an interactive fraud detection dashboard.
- Deploy the application for public access.

## ✨ Key Features

| Feature | Description |
|---|---|
| Exploratory Data Analysis | Visual analysis of transaction and fraud patterns |
| Feature Engineering | Time-based, customer-age, amount, and geographical features |
| Data Preprocessing | Numerical scaling and categorical encoding |
| Multiple ML Models | Logistic Regression, Random Forest, and XGBoost |
| Imbalance Handling | Class weighting and positive-class weighting |
| Threshold Optimization | Validation-based classification threshold selection |
| Model Evaluation | Precision, Recall, F1, ROC-AUC and PR-AUC |
| Fraud Analytics | Category-based and time-based fraud analysis |
| Individual Prediction | Predict fraud probability for a transaction |
| Bulk CSV Prediction | Analyze multiple transactions from uploaded files |
| Downloadable Reports | Export transaction-level prediction results |
| Cloud Deployment | Publicly accessible Streamlit application |

## 🛠️ Technology Stack

**Programming Language**
- Python

**Machine Learning**
- Scikit-learn
- XGBoost

**Data Processing**
- Pandas
- NumPy
- SciPy

**Data Visualization**
- Matplotlib

**Web Application**
- Streamlit

**Model Persistence**
- Joblib
- Pickle

**Development and Version Control**
- VS Code
- Git
- GitHub

## 📂 Dataset Description

The project uses two transaction datasets:

- `fraudTrain.csv` — Training dataset
- `fraudTest.csv` — Independent test dataset

### Dataset Statistics

| Dataset | Transactions | Columns |
|---|---:|---:|
| Training | 1,296,675 | 23 |
| Testing | 555,719 | 23 |

The datasets contain information including:

- Transaction date and time
- Transaction amount
- Merchant information
- Transaction category
- Customer demographic information
- Customer location
- Merchant geographical coordinates
- Fraud classification label

The target variable is `is_fraud`:

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

### Class Imbalance

Fraudulent transactions account for approximately 0.58% of the training dataset.

This imbalance makes accuracy alone insufficient for evaluating model performance.

Therefore, precision, recall, F1-score, ROC-AUC, and PR-AUC were used to compare the classification models.

**Dataset availability:** The original raw datasets are not included in this repository. They must be placed in the `data/` directory when reproducing the complete training pipeline.

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the distribution of transactions and identify potential fraud patterns.

The analysis covered:

1. Legitimate versus fraudulent transaction distribution.
2. Transaction amount distribution.
3. Fraud rates across transaction categories.
4. Fraud rates across different transaction hours.
5. Transaction volume patterns.

### Key Observations

- The dataset is highly imbalanced.
- Transaction amounts exhibit a right-skewed distribution.
- Fraud rates vary across transaction categories.
- Transaction hour provides potentially useful behavioral information.
- Temporal and geographical features contribute to transaction characterization.

These observations helped guide feature engineering and model development.

## ⚙️ Data Preprocessing and Feature Engineering

The preprocessing pipeline converts raw transaction data into a structured format suitable for machine learning.

### 1. Data Cleaning

- Removed unnecessary index columns.
- Parsed transaction timestamps.
- Parsed customer date-of-birth values.
- Excluded direct identifiers from model inputs.
- Checked missing values and duplicate records.

### 2. Feature Engineering

The following features were generated:

| Feature | Description |
|---|---|
| `transaction_hour` | Hour of transaction |
| `transaction_day` | Day of the month |
| `transaction_month` | Month of transaction |
| `day_of_week` | Day of the week |
| `is_weekend` | Indicates whether a transaction occurred on a weekend |
| `customer_age` | Customer age at transaction time |
| `log_amount` | Log-transformed transaction amount |
| `distance_km` | Distance between customer and merchant coordinates |

### 3. Encoding and Scaling

- One-Hot Encoding was used for categorical variables.
- StandardScaler was used for numerical features.
- Unknown categories were handled using `handle_unknown="ignore"`.
- The preprocessing pipeline was fitted on training data only.

### 4. Data Splitting

The training dataset was divided chronologically:

- 80% training data
- 20% validation data

The separate test dataset was reserved for final model evaluation.

This ensured that model selection and threshold optimization were performed without using the final test data.

## 🤖 Machine Learning Models

Three classification algorithms were developed and evaluated.

### 1. Logistic Regression

Used as the baseline classification model.

Configuration:
- Balanced class weights
- SAGA solver
- Maximum 500 iterations

### 2. Random Forest

An ensemble learning algorithm trained using a stratified sample of 250,000 transactions.

Configuration:
- 150 estimators
- Maximum tree depth of 18
- Balanced subsampling
- Parallel processing

### 3. XGBoost

XGBoost was selected as the final model because of its strong classification performance.

Configuration:
- 300 estimators
- Maximum depth of 6
- Learning rate of 0.05
- Subsample of 0.8
- Column sampling of 0.8
- Positive-class weighting
- Histogram-based tree construction

## 📊 Model Comparison

The following results were obtained on the validation dataset using the default classification threshold of 0.5.

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| Accuracy | 90.69% | 98.28% | **99.21%** |
| Precision | 4.69% | 22.94% | **42.37%** |
| Recall | 76.07% | 80.17% | **93.17%** |
| F1-Score | 0.0883 | 0.3567 | **0.5825** |
| ROC-AUC | 0.9063 | 0.9868 | **0.9966** |
| PR-AUC | 0.1649 | 0.7300 | **0.8805** |

### Why XGBoost?

XGBoost achieved the strongest overall results among the evaluated models.

Its high PR-AUC was particularly relevant because the dataset contains a very small proportion of fraudulent transactions.

Therefore, XGBoost was selected for integration into the final application.

## 🎚️ Threshold Optimization

A classification threshold determines the probability above which a transaction is classified as fraudulent.

The default threshold of 0.5 produced high recall but relatively low precision.

To improve the precision-recall balance, threshold optimization was performed using validation predictions.

Three strategies were evaluated:

| Strategy | Threshold | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Default | 0.5000 | 42.37% | 93.17% | 0.5825 |
| High Precision | 0.8395 | 70.00% | 86.48% | 0.7740 |
| Best F1 | 0.9551 | 87.67% | 79.06% | **0.8315** |

The Best F1 threshold of 0.9551 was selected using validation data and saved for use in the dashboard.

The High Precision strategy provides an alternative classification approach with a different balance between missed fraud and false alarms.

## 🧪 Final Test Evaluation

After model selection and threshold optimization, the selected model was evaluated on the separate test dataset.

### Final Test Performance

The following results correspond to the validation-selected Best F1 threshold of 0.9551.

| Metric | Test Result |
|---|---:|
| Accuracy | 99.84% |
| Precision | 81.49% |
| Recall | 75.71% |
| F1-Score | **0.7849** |
| ROC-AUC | **0.9947** |
| PR-AUC | **0.8311** |

### Test Threshold Comparison

| Strategy | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Default (0.5) | 33.67% | 91.24% | 0.4918 |
| Best F1 (0.9551) | **81.49%** | 75.71% | **0.7849** |
| High Precision (0.8395) | 61.07% | 83.59% | 0.7058 |

### Interpretation

The Best F1 strategy substantially improves precision compared with the default threshold.

However, it also reduces recall, meaning some fraudulent transactions are missed.

The results demonstrate the importance of selecting a classification threshold according to the intended fraud detection requirements.

The final test dataset was not used to select the threshold.

## 🖥️ FraudGuard Interactive Dashboard

The trained model is integrated into a publicly accessible Streamlit application.

### Live Application

🚀 **https://fraudsguard.streamlit.app/**

### Dashboard Modules

**1. Overview**

- Dataset statistics
- Legitimate and fraudulent transaction distribution
- Selected model information
- Test precision and recall
- F1-score and selected threshold

**2. Fraud Analytics**

- Fraud rate by transaction category
- Fraud rate by transaction hour
- Transaction volume by hour
- Transaction amount distribution
- Fraud and legitimate transaction statistics
- Category risk summary

**3. Model Performance**

- Model evaluation metrics
- Threshold strategy comparison
- Confusion matrices
- Classification error analysis

**4. Transaction Prediction**

- Enter transaction information through an interactive form.
- Generate fraud probability.
- Select a classification strategy.
- Display the resulting transaction classification.

**5. CSV Prediction**

- Upload transaction datasets.
- Validate required input columns.
- Process multiple transactions.
- Display fraud and legitimate transaction counts.
- Filter prediction results.
- Download the complete prediction report.

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── dashboard/
│   ├── app.py
│   └── utils.py
│
├── data/
│   ├── fraudTrain.csv
│   ├── fraudTest.csv
│   └── sample_transactions.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   ├── preprocessor.pkl
│   └── threshold_settings.pkl
│
├── notebooks/
│   └── 01_eda.py
│
├── outputs/
│   ├── class_distribution.png
│   ├── transaction_amount.png
│   ├── fraud_by_category.png
│   ├── fraud_by_hour.png
│   ├── model_results.csv
│   ├── threshold_results.csv
│   ├── threshold_analysis.png
│   ├── test_evaluation_results.csv
│   ├── test_auc_results.csv
│   ├── test_metrics_comparison.png
│   ├── test_confusion_matrices.png
│   ├── error_comparison.png
│   └── train_processed.csv.gz
│
├── src/
│   ├── preprocessing.py
│   ├── prepare_data.py
│   ├── train_models.py
│   ├── random_forest.py
│   ├── xgboost_model.py
│   ├── threshold_tuning.py
│   ├── evaluate_test.py
│   └── visualize_results.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Installation and Local Setup

### Prerequisites

- Python 3.13
- Git
- pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/devanshikhandelwal/Credit-Card-Fraud-Detection.git
```

### Step 2: Navigate to the Project

```bash
cd Credit-Card-Fraud-Detection
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Dashboard

```bash
python -m streamlit run dashboard/app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

### Reproducing the Training Pipeline

To reproduce the machine learning workflow from scratch:

1. Place the original `fraudTrain.csv` and `fraudTest.csv` files inside the `data/` directory.
2. Run the preprocessing script.
3. Prepare the training and validation datasets.
4. Train the classification models.
5. Perform threshold optimization.
6. Evaluate the selected model on the independent test dataset.
7. Generate evaluation visualizations.

The trained models and preprocessing artifacts are already included in the repository, so the deployed dashboard does not require retraining.

## 🔮 Future Improvements

Potential improvements include:

- Adding SHAP-based model explainability.
- Developing a REST API for model predictions.
- Integrating real-time transaction streams.
- Experimenting with additional anomaly detection algorithms.
- Introducing automated model monitoring.
- Detecting changes in transaction distributions and model performance.
- Exploring cost-sensitive threshold optimization.
- Adding authentication and access control for a production-oriented application.

## ⚠️ Limitations and Responsible Use

FraudGuard is an academic machine learning prototype developed using a historical transaction dataset.

It is not connected to a live banking system, payment gateway, or financial institution.

Its predictions should not be treated as definitive financial decisions.

Performance on historical data does not guarantee performance on future transactions. A real-world fraud detection system would require updated training data, ongoing validation, monitoring, security controls, and human oversight.

## 👩‍💻 Author

**Devanshi Khandelwal**

B.Tech — Artificial Intelligence and Machine Learning

- **GitHub:** https://github.com/devanshikhandelwal
- **LinkedIn:** https://linkedin.com/in/devanshi-khandelwal
- **Project Repository:** https://github.com/devanshikhandelwal/Credit-Card-Fraud-Detection
- **Live Application:** https://fraudsguard.streamlit.app/

---

## ⭐ Final Note

FraudGuard demonstrates the application of machine learning to an important financial security problem.

From exploratory data analysis and feature engineering to model comparison, threshold optimization, independent testing, and cloud deployment, this project brings together the complete machine learning development workflow.

**Built with Python. Powered by XGBoost. Deployed with Streamlit.**

*Turning transaction data into actionable fraud detection insights.*