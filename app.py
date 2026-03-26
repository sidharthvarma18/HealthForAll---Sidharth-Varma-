
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(layout="wide")

st.title("🚀 FitFounder AI Dashboard (Stable Version)")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("fitness_app_synthetic_dataset_2000.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ---------------- TARGET ----------------
target_col = "Q20_Likelihood_to_Use_App"

df["target"] = df[target_col].apply(
    lambda x: 1 if str(x).strip() in ["Likely", "Very likely"] else 0
)

X = df.drop(columns=[target_col, "target"], errors="ignore")
y = df["target"]

# ---------------- PREPROCESS ----------------
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols),

    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), num_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=150, random_state=42))
])

# ---------------- TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader("🤖 Classification Metrics")
st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
st.write("Precision:", round(precision_score(y_test, y_pred), 3))
st.write("Recall:", round(recall_score(y_test, y_pred), 3))
st.write("F1 Score:", round(f1_score(y_test, y_pred), 3))

# ---------------- UPLOAD ----------------
st.subheader("📥 Upload New Customers")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    new_df = pd.read_csv(uploaded)

    # Align columns safely
    for col in X.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan

    new_df = new_df[X.columns]

    preds = model.predict(new_df)
    probs = model.predict_proba(new_df)[:,1]

    new_df["Prediction"] = preds
    new_df["Probability"] = probs

    st.success("✅ Predictions generated successfully")
    st.dataframe(new_df.head())

st.success("🔥 App running without errors")
