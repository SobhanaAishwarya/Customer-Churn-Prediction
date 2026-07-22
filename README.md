# Customer Churn Prediction

A Streamlit dashboard that predicts whether a telecom customer is likely to
churn. Upload customer data and get live KPIs, business-analytics charts,
class-balancing with SMOTE, and a side-by-side comparison of three
classification models — all rendered on the fly.

## Features

- **CSV/Excel upload** — or falls back to the bundled `Telco_Customer_Churn.csv`
  dataset
- **KPI cards** — total customers, churned customers, retention rate, churn rate
- **Business analytics** — churn distribution, contract type vs. churn, tenure
  analysis, monthly charges vs. churn
- **Preprocessing** — missing-value handling, categorical encoding, feature
  scaling
- **Class balancing** — SMOTE oversampling to correct for imbalanced churn labels
- **Model comparison** — Logistic Regression, Decision Tree, and Neural Network
  (MLP), each with accuracy scores and confusion matrices
- **Excel export** — download per-customer churn predictions and probabilities

## Tech stack

Python, Streamlit, Pandas, NumPy, Scikit-learn, imbalanced-learn (SMOTE),
Matplotlib, Seaborn

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Upload your own CSV/Excel file, or leave it
empty to explore the bundled Telco dataset.

## Dataset

`Telco_Customer_Churn.csv` (also provided as `Dataset_CCP.csv`) — customer
records including tenure, contract type, monthly/total charges, payment
method, and the churn label (Yes/No).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard — the entire pipeline runs on each upload |
| `customer_churn_prediction.ipynb` | Exploratory notebook version of the analysis |
| `customer_churn_prediction.pdf` | Notebook exported as a report |
| `Telco_Customer_Churn.csv` / `Dataset_CCP.csv` | Sample/default dataset |
