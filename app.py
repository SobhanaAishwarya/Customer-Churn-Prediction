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

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide",
)

# ==========================
# THEME
# ==========================

NAVY = "#0F1B2D"
SLATE = "#334155"
SLATE_LIGHT = "#64748B"
ACCENT = "#2563EB"
GOLD = "#C8973C"
LINE = "#E4E7EC"
PALETTE = [ACCENT, GOLD, NAVY, SLATE_LIGHT]

st.markdown(
    """
    <style>
    .stApp { background: #F5F7FA; font-family: 'Inter', 'Segoe UI', sans-serif; }
    .block-container { padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1220px; }
    h1, h2, h3 { color: #0F1B2D; font-weight: 700; }

    .app-header { margin-bottom: 6px; }
    .app-header .eyebrow {
        font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #2563EB; margin-bottom: 4px;
    }
    .app-header h1 { font-size: 26px; margin: 0; }
    .app-header p { font-size: 14px; color: #64748B; margin: 4px 0 0; }

    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 2px 10px rgba(15,27,45,0.06);
    }
    [data-testid="stMetricLabel"] { color: #64748B; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #0F1B2D; font-weight: 800; }

    .section-title {
        font-size: 12.5px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.06em; color: #64748B; margin: 30px 0 14px;
    }

    .chart-card {
        background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 14px;
        padding: 18px 20px 6px; box-shadow: 0 2px 10px rgba(15,27,45,0.06);
        margin-bottom: 18px;
    }
    .chart-card h4 {
        margin: 0 0 8px; font-size: 13px; font-weight: 700; color: #0F1B2D;
    }

    .insight-card {
        background: #FFFFFF; border: 1px solid #E4E7EC; border-left: 4px solid #2563EB;
        border-radius: 12px; padding: 18px 22px; box-shadow: 0 2px 10px rgba(15,27,45,0.06);
    }
    .insight-card ul { margin: 0; padding-left: 18px; }
    .insight-card li { color: #334155; font-size: 14px; margin-bottom: 8px; line-height: 1.5; }

    [data-testid="stDataFrame"] { border: 1px solid #E4E7EC; border-radius: 12px; }
    [data-testid="stFileUploader"] {
        background: #FFFFFF; border: 1.5px dashed #2563EB; border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": LINE,
    "axes.labelcolor": SLATE,
    "text.color": NAVY,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "axes.grid": True,
    "grid.color": "#EEF1F5",
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})
sns.set_style("white")
sns.set_palette(PALETTE)


def chart_card_open(title: str) -> None:
    st.markdown(f'<div class="chart-card"><h4>{title}</h4>', unsafe_allow_html=True)


def chart_card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="app-header">
      <div class="eyebrow">Customer Analytics</div>
      <h1>Customer Churn Prediction Dashboard</h1>
      <p>Identify at-risk customers and the business factors driving churn.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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

st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
st.dataframe(df.head(), use_container_width=True)

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

st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)

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

st.markdown('<div class="section-title">Business Analytics</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:

    chart_card_open("Customer Churn Distribution")

    churn_counts = df["Churn"].value_counts()
    color_map = {"No": NAVY, "Yes": GOLD}
    pie_colors = [color_map.get(str(label), ACCENT) for label in churn_counts.index]

    fig1, ax1 = plt.subplots(figsize=(4.3, 4.3), dpi=140)

    wedges, _, autotexts = ax1.pie(
        churn_counts.values,
        labels=churn_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=pie_colors,
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
        textprops={"fontsize": 10.5, "color": SLATE},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax1.set_ylabel("")
    fig1.tight_layout()

    st.pyplot(fig1, use_container_width=False)

    chart_card_close()

with c2:

    if "Contract" in df.columns:

        chart_card_open("Contract Type vs Churn")

        fig2, ax2 = plt.subplots(figsize=(6, 4.6), dpi=140)

        sns.countplot(
            x="Contract",
            hue="Churn",
            data=df,
            ax=ax2,
            palette=[NAVY, GOLD],
        )

        ax2.set_xlabel("")
        ax2.set_ylabel("Customers")
        ax2.tick_params(axis="x", rotation=15)
        ax2.legend(title="Churn", frameon=False)
        fig2.tight_layout()

        st.pyplot(fig2, use_container_width=True)

        chart_card_close()

# ==========================
# TENURE ANALYSIS
# ==========================

if "tenure" in df.columns:

    st.markdown('<div class="section-title">Tenure Analysis</div>', unsafe_allow_html=True)

    chart_card_open("Customer Tenure Distribution by Churn")

    fig3, ax3 = plt.subplots(figsize=(11, 4.4), dpi=140)

    sns.histplot(
        data=df,
        x="tenure",
        hue="Churn",
        bins=20,
        ax=ax3,
        palette=[NAVY, GOLD],
    )

    ax3.set_xlabel("Tenure (months)")
    ax3.set_ylabel("Customers")
    fig3.tight_layout()

    st.pyplot(fig3, use_container_width=True)

    chart_card_close()

# ==========================
# MONTHLY CHARGES ANALYSIS
# ==========================

if "MonthlyCharges" in df.columns:

    st.markdown('<div class="section-title">Monthly Charges Analysis</div>', unsafe_allow_html=True)

    chart_card_open("Monthly Charges by Churn Status")

    fig4, ax4 = plt.subplots(figsize=(11, 4.4), dpi=140)

    sns.boxplot(
        x="Churn",
        y="MonthlyCharges",
        data=df,
        ax=ax4,
        palette=[NAVY, GOLD],
    )

    ax4.set_xlabel("")
    ax4.set_ylabel("Monthly Charges ($)")
    fig4.tight_layout()

    st.pyplot(fig4, use_container_width=True)

    chart_card_close()

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
# CLASS DISTRIBUTION: BEFORE / AFTER SMOTE
# ==========================

st.markdown('<div class="section-title">Class Distribution — Before vs After SMOTE</div>', unsafe_allow_html=True)

before_counts = pd.DataFrame({
    "Class": ["No", "Yes"],
    "Count": [int((y == 0).sum()), int((y == 1).sum())],
})

after_counts = pd.DataFrame({
    "Class": ["No", "Yes"],
    "Count": [int((y_resampled == 0).sum()), int((y_resampled == 1).sum())],
})

before_col, after_col = st.columns(2)

with before_col:
    st.dataframe(before_counts, use_container_width=True, hide_index=True)

    chart_card_open("Before SMOTE")

    fig_before, ax_before = plt.subplots(figsize=(5.6, 4.2), dpi=140)

    sns.barplot(
        x="Class",
        y="Count",
        data=before_counts,
        ax=ax_before,
        palette=[NAVY, GOLD],
    )

    ax_before.set_xlabel("")
    ax_before.set_ylabel("Customers")
    for container in ax_before.containers:
        ax_before.bar_label(container, fmt="%d", padding=3, fontsize=10, color=SLATE)
    fig_before.tight_layout()

    st.pyplot(fig_before, use_container_width=True)

    chart_card_close()

with after_col:
    st.dataframe(after_counts, use_container_width=True, hide_index=True)

    chart_card_open("After SMOTE")

    fig_after, ax_after = plt.subplots(figsize=(5.6, 4.2), dpi=140)

    sns.barplot(
        x="Class",
        y="Count",
        data=after_counts,
        ax=ax_after,
        palette=[NAVY, GOLD],
    )

    ax_after.set_xlabel("")
    ax_after.set_ylabel("Customers")
    for container in ax_after.containers:
        ax_after.bar_label(container, fmt="%d", padding=3, fontsize=10, color=SLATE)
    fig_after.tight_layout()

    st.pyplot(fig_after, use_container_width=True)

    chart_card_close()

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

st.markdown('<div class="section-title">Model Accuracy Comparison</div>', unsafe_allow_html=True)

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

st.dataframe(accuracy_df, use_container_width=True, hide_index=True)

chart_card_open("Accuracy by Model")

fig5, ax5 = plt.subplots(figsize=(11, 4.2), dpi=140)

sns.barplot(
    x="Model",
    y="Accuracy",
    data=accuracy_df,
    ax=ax5,
    palette=PALETTE[:3],
)

ax5.set_ylim(0, 1)
ax5.set_xlabel("")
fig5.tight_layout()

st.pyplot(fig5, use_container_width=True)

chart_card_close()

# ==========================
# ALL CONFUSION MATRICES
# ==========================

st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:

    chart_card_open("Logistic Regression")

    fig, ax = plt.subplots(figsize=(4, 3.6), dpi=140)

    sns.heatmap(
        confusion_matrix(
            y_test,
            lr_pred
        ),
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        cbar=False,
    )
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    chart_card_close()

with c2:

    chart_card_open("Decision Tree")

    fig, ax = plt.subplots(figsize=(4, 3.6), dpi=140)

    sns.heatmap(
        confusion_matrix(
            y_test,
            dt_pred
        ),
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        cbar=False,
    )
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    chart_card_close()

with c3:

    chart_card_open("Neural Network")

    fig, ax = plt.subplots(figsize=(4, 3.6), dpi=140)

    sns.heatmap(
        confusion_matrix(
            y_test,
            mlp_pred
        ),
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        cbar=False,
    )
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    chart_card_close()

# ==========================
# EXCEL REPORT EXPORT
# ==========================

st.markdown('<div class="section-title">Download Prediction Report</div>', unsafe_allow_html=True)

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
    label="Download Excel Report",
    data=buffer.getvalue(),
    file_name="Customer_Churn_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ==========================
# KEY INSIGHTS
# ==========================

st.markdown('<div class="section-title">Business Insights</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="insight-card">
      <ul>
        <li>Customers with shorter tenure are more likely to churn.</li>
        <li>Month-to-month contracts show higher churn.</li>
        <li>Customers with higher monthly charges tend to churn more.</li>
        <li>Neural Network achieved the highest accuracy.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
