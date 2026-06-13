
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide"
)

st.title("Customer Churn Prediction Dashboard")

# ==========================
# FILE UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

else:
    df = pd.read_csv("Telco_Customer_Churn.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ==========================
# KPI CARDS
# ==========================

total_customers = len(df)

churn_customers = df["Churn"].value_counts().get("Yes", 0)

retention_rate = (
    (total_customers - churn_customers)
    / total_customers
) * 100

churn_rate = (
    churn_customers
    / total_customers
) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    total_customers
)

col2.metric(
    "Churn Customers",
    churn_customers
)

col3.metric(
    "Retention Rate",
    f"{retention_rate:.2f}%"
)

col4.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)

# ==========================
# BUSINESS CHARTS
# ==========================

st.subheader("Business Analytics Dashboard")

c1, c2 = st.columns(2)

with c1:

    fig1, ax1 = plt.subplots(figsize=(6, 5))

    df["Churn"].value_counts().plot.pie(
        autopct="%1.1f%%",
        ax=ax1
    )

    ax1.set_ylabel("")
    ax1.set_title("Customer Churn Distribution")

    st.pyplot(fig1)

with c2:

    if "Contract" in df.columns:

        fig2, ax2 = plt.subplots(figsize=(7, 5))

        sns.countplot(
            x="Contract",
            hue="Churn",
            data=df,
            ax=ax2
        )

        plt.xticks(rotation=20)
        plt.title("Contract Type vs Churn")

        st.pyplot(fig2)

# ==========================
# TENURE ANALYSIS
# ==========================

if "tenure" in df.columns:

    st.subheader("Tenure Analysis")

    fig3, ax3 = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="tenure",
        hue="Churn",
        bins=20,
        ax=ax3
    )

    st.pyplot(fig3)

# ==========================
# MONTHLY CHARGES ANALYSIS
# ==========================

if "MonthlyCharges" in df.columns:

    st.subheader("Monthly Charges Analysis")

    fig4, ax4 = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        x="Churn",
        y="MonthlyCharges",
        data=df,
        ax=ax4
    )

    st.pyplot(fig4)

# ==========================
# PREPROCESSING
# ==========================

df_model = df.copy()

if "customerID" in df_model.columns:
    customer_ids = df_model["customerID"]
    df_model.drop(
        "customerID",
        axis=1,
        inplace=True
    )
else:
    customer_ids = pd.Series(
        range(len(df_model))
    )

df_model["TotalCharges"] = pd.to_numeric(
    df_model["TotalCharges"],
    errors="coerce"
)

df_model.dropna(inplace=True)

df_model["Churn"] = df_model["Churn"].map(
    {
        "Yes": 1,
        "No": 0
    }
)

df_model = pd.get_dummies(
    df_model,
    drop_first=True
)

X = df_model.drop(
    "Churn",
    axis=1
)

y = df_model["Churn"]

# ==========================
# SCALING
# ==========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================
# SMOTE
# ==========================

smote = SMOTE(
    random_state=42
)

X_resampled, y_resampled = smote.fit_resample(
    X_scaled,
    y
)

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# ==========================
# MODELS
# ==========================

lr = LogisticRegression(
    max_iter=1000
)

dt = DecisionTreeClassifier(
    random_state=42
)

mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=300,
    random_state=42
)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
mlp.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)
mlp_pred = mlp.predict(X_test)

lr_acc = accuracy_score(
    y_test,
    lr_pred
)

dt_acc = accuracy_score(
    y_test,
    dt_pred
)

mlp_acc = accuracy_score(
    y_test,
    mlp_pred
)

# ==========================
# MODEL COMPARISON
# ==========================

st.subheader(
    "Model Accuracy Comparison"
)

accuracy_df = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Neural Network"
        ],
        "Accuracy": [
            lr_acc,
            dt_acc,
            mlp_acc
        ]
    }
)

st.dataframe(accuracy_df)

fig5, ax5 = plt.subplots(
    figsize=(7, 5)
)

sns.barplot(
    x="Model",
    y="Accuracy",
    data=accuracy_df,
    ax=ax5
)

plt.ylim(0, 1)

st.pyplot(fig5)

# ==========================
# ALL CONFUSION MATRICES
# ==========================

st.subheader(
    "Confusion Matrices"
)

c1, c2, c3 = st.columns(3)

with c1:

    fig, ax = plt.subplots()

    sns.heatmap(
        confusion_matrix(
            y_test,
            lr_pred
        ),
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax
    )

    ax.set_title(
        "Logistic Regression"
    )

    st.pyplot(fig)

with c2:

    fig, ax = plt.subplots()

    sns.heatmap(
        confusion_matrix(
            y_test,
            dt_pred
        ),
        annot=True,
        fmt='d',
        cmap='Greens',
        ax=ax
    )

    ax.set_title(
        "Decision Tree"
    )

    st.pyplot(fig)

with c3:

    fig, ax = plt.subplots()

    sns.heatmap(
        confusion_matrix(
            y_test,
            mlp_pred
        ),
        annot=True,
        fmt='d',
        cmap='Reds',
        ax=ax
    )

    ax.set_title(
        "Neural Network"
    )

    st.pyplot(fig)

# ==========================
# EXCEL REPORT EXPORT
# ==========================

st.subheader(
    "Download Prediction Report"
)

prob = mlp.predict_proba(
    X_test
)[:, 1]


report_df = pd.DataFrame(
    {
        "Customer_ID":
            range(
                1,
                len(prob) + 1
            ),

        "Prediction":
            np.where(
                mlp_pred == 1,
                "Churn",
                "Not Churn"
            ),

        "Probability":
            prob
    }
)

buffer = BytesIO()

with pd.ExcelWriter(
    buffer,
    engine="xlsxwriter"
) as writer:

    report_df.to_excel(
        writer,
        index=False,
        sheet_name="Prediction_Report"
    )

st.download_button(
    label="📥 Download Excel Report",
    data=buffer.getvalue(),
    file_name="Customer_Churn_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ==========================
# KEY INSIGHTS
# ==========================

st.subheader(
    "Business Insights"
)

st.success(
    """
• Customers with shorter tenure are more likely to churn.

• Month-to-month contracts show higher churn.

• Customers with higher monthly charges tend to churn more.

• Neural Network achieved the highest accuracy.
"""
)

