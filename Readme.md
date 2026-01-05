# BankSim Fraud Detection Analysis

## Overview
We have implements a comprehensive machine learning pipeline for fraud detection on the BankSim synthetic transaction dataset. The code focuses on data loading, preprocessing, exploratory analysis, model training (supervised and unsupervised), evaluation, and optimization.

## Key Components
- **Data Loading & Exploration**: Loads CSV data and performs initial stats, info, and distribution checks.
- **EDA**: Generates visualizations for distributions, correlations, fraud patterns, and category analysis.
- **Preprocessing**: Handles categorical encoding (LabelEncoder), scaling (StandardScaler), outlier detection, and imbalanced sampling (SMOTE, under-sampling).
- **Feature Engineering**: Creates PCA/t-SNE reductions and engineered features (e.g., amount bins, time-based).
- **Models**: Trains multiple classifiers:
  - Supervised: Random Forest, Gradient Boosting, SVM, Logistic Regression, XGBoost, LightGBM, CatBoost.
  - Unsupervised: Isolation Forest, Local Outlier Factor.
- **Evaluation**: Computes metrics (precision, recall, F1, ROC-AUC, PR-AUC) with cross-validation, confusion matrices, and curve plots.
- **Optimization**: Grid search, threshold tuning via Youden's J, and model comparison.
- **Visualization**: ROC/PR curves, feature importance, and performance bar charts using Matplotlib/Seaborn.

## Requirements
- Python 3.12+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost, lightgbm, catboost.
- Install via: `pip install catboost` (handled in notebook).

## Usage
1. Upload `bs140513_032310.csv` to Colab or local environment.
2. Run cells sequentially in Jupyter/Colab.
3. Outputs: Plots, metrics tables, and model artifacts (no saving; in-memory).

## Notes
- Code is modular; sections are self-contained for easy modification.
- Warnings suppressed; GPU optional (for CatBoost/XGBoost).
- Dataset: ~595K rows, 10 columns (step, customer, age, etc.).