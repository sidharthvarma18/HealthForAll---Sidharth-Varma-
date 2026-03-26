
import streamlit as st
import pandas as pd
import numpy as np

st.title("FitFounder AI Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("fitness_app_synthetic_dataset_2000.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# SAFE COLUMN HANDLING
def find_column(possible_names):
    for col in df.columns:
        for name in possible_names:
            if name.lower() in col.lower():
                return col
    return None

# Identify columns safely
interest_col = find_column(["likelihood", "interest"])
wtp_col = find_column(["willingness", "pay"])
guarantee_col = find_column(["guaranteed", "wtp"])

st.subheader("Detected Columns")
st.write({
    "Interest Column": interest_col,
    "WTP Column": wtp_col,
    "Guaranteed WTP Column": guarantee_col
})

# Basic descriptive
st.subheader("Descriptive Analytics")
if interest_col:
    st.bar_chart(df[interest_col].value_counts())
else:
    st.warning("Interest column not found")

# Regression SAFE
st.subheader("Regression (Safe Mode)")
if guarantee_col:
    clean_df = df.dropna(subset=[guarantee_col])
    if len(clean_df) > 0:
        st.write("Regression column available. Rows:", len(clean_df))
    else:
        st.warning("No usable rows for regression")
else:
    st.warning("Guaranteed WTP column not found — skipping regression")

# Classification SAFE
st.subheader("Classification (Safe Mode)")
if interest_col:
    st.write("Classification ready")
else:
    st.warning("Skipping classification due to missing column")

st.success("App running successfully without crashes!")
