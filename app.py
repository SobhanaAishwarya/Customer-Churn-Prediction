import streamlit as st
import pandas as pd

st.title("Customer Churn Prediction System")

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Columns")
st.write(df.columns)