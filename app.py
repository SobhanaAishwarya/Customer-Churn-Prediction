import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

st.title("Customer Churn Prediction System")

# Load Dataset
df = pd.read_csv("Telco_Customer_Churn.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# Data Preprocessing
df.drop(['customerID'], axis=1, inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# Models
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)
mlp_pred = mlp.predict(X_test)

# Accuracy
lr_acc = accuracy_score(y_test, lr_pred)
dt_acc = accuracy_score(y_test, dt_pred)
mlp_acc = accuracy_score(y_test, mlp_pred)

st.subheader("Model Accuracy")

accuracy_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Neural Network"],
    "Accuracy": [lr_acc, dt_acc, mlp_acc]
})

st.write(accuracy_df)

# Accuracy Chart
fig, ax = plt.subplots(figsize=(7,5))
sns.barplot(x="Model", y="Accuracy", data=accuracy_df, ax=ax)
plt.ylim(0,1)

st.pyplot(fig)

# Confusion Matrix
st.subheader("Neural Network Confusion Matrix")

cm = confusion_matrix(y_test, mlp_pred)

fig2, ax2 = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)

st.pyplot(fig2)

# Churn Distribution
st.subheader("Customer Churn Distribution")

labels = ['Not Churned', 'Churned']
sizes = y.value_counts()

fig3, ax3 = plt.subplots(figsize=(5,5))
ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

st.pyplot(fig3)

st.success("Customer Churn Prediction Project Executed Successfully")