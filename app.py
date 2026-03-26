
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans

st.set_page_config(page_title="FitFounder AI", layout="wide")

st.title("🚀 FitFounder AI - Full Dashboard")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("fitness_app_synthetic_dataset_2000.csv")
    return df

df = load_data()

# ---------------- COLUMN DETECTION ----------------
def find_col(keywords):
    for col in df.columns:
        for k in keywords:
            if k.lower() in col.lower():
                return col
    return None

interest_col = find_col(["likelihood"])
wtp_col = find_col(["willingness"])
guarantee_col = find_col(["guaranteed"])

# ---------------- DESCRIPTIVE ----------------
st.header("📊 Market Overview")
st.dataframe(df.head())

if interest_col:
    st.subheader("Interest Distribution")
    st.bar_chart(df[interest_col].value_counts())

# ---------------- PREPROCESS ----------------
df_model = df.copy()

# encode all categorical
le_dict = {}
for col in df_model.columns:
    if df_model[col].dtype == 'object':
        le = LabelEncoder()
        df_model[col] = df_model[col].astype(str)
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le

# ---------------- CLASSIFICATION ----------------
st.header("🤖 Classification Model")

if interest_col:
    y = df_model[interest_col]

    # Binary conversion
    y = y.apply(lambda x: 1 if x >= 3 else 0)

    X = df_model.drop(columns=[interest_col], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1:", f1_score(y_test, y_pred))
    st.write("ROC AUC:", roc_auc_score(y_test, y_prob))

# ---------------- CLUSTERING ----------------
st.header("🧩 Customer Segmentation")

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(df_model)

df['Cluster'] = clusters
st.write(df['Cluster'].value_counts())

# ---------------- REGRESSION ----------------
st.header("💰 Regression Model")

if wtp_col:
    reg_df = df_model.dropna(subset=[wtp_col])

    if len(reg_df) > 0:
        y_reg = reg_df[wtp_col]
        X_reg = reg_df.drop(columns=[wtp_col], errors='ignore')

        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2)

        reg = RandomForestRegressor()
        reg.fit(X_train, y_train)

        pred = reg.predict(X_test)
        st.write("Regression running successfully")

    else:
        st.warning("No data for regression")

# ---------------- UPLOAD ----------------
st.header("📥 Upload New Customers")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    new_df = pd.read_csv(uploaded)

    for col in new_df.columns:
        if col in le_dict:
            new_df[col] = new_df[col].astype(str)
            new_df[col] = le_dict[col].transform(new_df[col])

    preds = clf.predict(new_df)
    probs = clf.predict_proba(new_df)[:,1]

    new_df["Predicted_Interest"] = preds
    new_df["Probability"] = probs

    st.write(new_df.head())

st.success("🔥 FULL APP RUNNING")
