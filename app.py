
import io
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    px = None
    go = None
    PLOTLY_AVAILABLE = False

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from mlxtend.frequent_patterns import apriori, association_rules


st.set_page_config(page_title="FitFounder Intelligence Hub", layout="wide")

DATA_FILE_CANDIDATES = ["fitness_app_synthetic_dataset_2000.csv", "fitness_app_synthetic_dataset_2000_ml_ready.csv"]

MULTI_SELECT_COLS = ["Q17_Preferred_Features", "Q19_What_Would_Make_You_Pay"]

FEATURE_OPTIONS = [
    "AI workout plan", "Form correction via video", "Diet planning",
    "Calorie tracking via photo", "Travel-friendly plans", "Allergy-specific diets"
]
PAY_REASON_OPTIONS = [
    "Proven results", "Personalization", "Expert guidance",
    "Convenience", "All-in-one solution"
]

LIKELIHOOD_MAP = {
    "Very unlikely": 0, "Unlikely": 0, "Neutral": 0,
    "Likely": 1, "Very likely": 1
}
WTP_MAP = {
    "Free only": 0, "₹99-₹199": 149, "₹199-₹499": 349, "₹499-₹999": 749, "₹999+": 1200
}
GUARANTEED_WTP_MAP = {
    "Nothing": 0, "₹199": 199, "₹499": 499, "₹999": 999, "₹2000+": 2000
}
ORDER_MAPS = {
    "Q4_Monthly_Income": ["<20,000", "20,000-50,000", "50,000-1,00,000", "1,00,000-2,00,000", "2,00,000+"],
    "Q5_Exercise_Frequency": ["Never", "1-2 times/week", "3-4 times/week", "5+ times/week"],
    "Q13_Travel_Frequency": ["Rarely", "Occasionally", "Frequently"],
    "Q14_Daily_Routine": ["Very structured", "Somewhat structured", "Highly unpredictable"],
    "Q15_Time_for_Fitness": ["<30 mins", "30-60 mins", "60-90 mins", "90+ mins"],
    "Q18_Willingness_to_Pay": ["Free only", "₹99-₹199", "₹199-₹499", "₹499-₹999", "₹999+"],
    "Q20_Likelihood_to_Use_App": ["Very unlikely", "Unlikely", "Neutral", "Likely", "Very likely"],
    "Q23_Days_Followed_Plan_Last_30_Days": ["0-5", "6-10", "11-20", "20+"],
    "Q26_Guaranteed_Results_WTP": ["Nothing", "₹199", "₹499", "₹999", "₹2000+"],
    "Q28_Meal_Control_Level": ["Full control (cook myself)", "Partial control", "Very limited control (hostel/mess)"],
}


REQUIRED_RAW_COLUMNS = [
    "Respondent_ID", "Q1_Age_Group","Q2_City_Type","Q3_Occupation","Q4_Monthly_Income",
    "Q5_Exercise_Frequency","Q6_Workout_Location","Q7_Primary_Fitness_Goal",
    "Q8_Biggest_Fitness_Challenge","Q9_Diet_Type","Q10_Dietary_Restrictions",
    "Q11_Meal_Management","Q12_Biggest_Diet_Challenge","Q13_Travel_Frequency",
    "Q14_Daily_Routine","Q15_Time_for_Fitness","Q16_Current_Fitness_App_Usage",
    "Q17_Preferred_Features","Q18_Willingness_to_Pay","Q19_What_Would_Make_You_Pay",
    "Q20_Likelihood_to_Use_App","Q21_Current_Weight_Category","Q22_Problem_Intensity_Score",
    "Q23_Days_Followed_Plan_Last_30_Days","Q24_When_Stop_Routine",
    "Q25_Why_Stopped_Fitness_Apps","Q26_Guaranteed_Results_WTP","Q27_Main_Motivation",
    "Q28_Meal_Control_Level"
]

COLUMN_ALIASES = {
    "q1_age_group": "Q1_Age_Group",
    "q2_city_type": "Q2_City_Type",
    "q3_occupation": "Q3_Occupation",
    "q4_monthly_income": "Q4_Monthly_Income",
    "q5_exercise_frequency": "Q5_Exercise_Frequency",
    "q6_workout_location": "Q6_Workout_Location",
    "q7_primary_fitness_goal": "Q7_Primary_Fitness_Goal",
    "q8_biggest_fitness_challenge": "Q8_Biggest_Fitness_Challenge",
    "q9_diet_type": "Q9_Diet_Type",
    "q10_dietary_restrictions": "Q10_Dietary_Restrictions",
    "q11_meal_management": "Q11_Meal_Management",
    "q12_biggest_diet_challenge": "Q12_Biggest_Diet_Challenge",
    "q13_travel_frequency": "Q13_Travel_Frequency",
    "q14_daily_routine": "Q14_Daily_Routine",
    "q15_time_for_fitness": "Q15_Time_for_Fitness",
    "q16_current_fitness_app_usage": "Q16_Current_Fitness_App_Usage",
    "q17_preferred_features": "Q17_Preferred_Features",
    "q18_willingness_to_pay": "Q18_Willingness_to_Pay",
    "q19_what_would_make_you_pay": "Q19_What_Would_Make_You_Pay",
    "q20_likelihood_to_use_app": "Q20_Likelihood_to_Use_App",
    "q21_current_weight_category": "Q21_Current_Weight_Category",
    "q22_problem_intensity_score": "Q22_Problem_Intensity_Score",
    "q23_days_followed_plan_last_30_days": "Q23_Days_Followed_Plan_Last_30_Days",
    "q24_when_stop_routine": "Q24_When_Stop_Routine",
    "q25_why_stopped_fitness_apps": "Q25_Why_Stopped_Fitness_Apps",
    "q26_guaranteed_results_wtp": "Q26_Guaranteed_Results_WTP",
    "q27_main_motivation": "Q27_Main_Motivation",
    "q28_meal_control_level": "Q28_Meal_Control_Level",
    "respondent_id": "Respondent_ID",
    "persona_segment": "Persona_Segment"
}

def _normalize_key(name):
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(name)).strip('_')

def normalize_columns(df):
    rename_map = {}
    for col in df.columns:
        norm = _normalize_key(col)
        if norm in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[norm]
    out = df.rename(columns=rename_map).copy()
    return out

def ensure_required_columns(df):
    out = df.copy()
    for col in REQUIRED_RAW_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    if "Respondent_ID" not in out.columns:
        out.insert(0, "Respondent_ID", [f"R_{i+1:05d}" for i in range(len(out))])
    return out

BASE_FEATURES = [
    "Q1_Age_Group","Q2_City_Type","Q3_Occupation","Q4_Monthly_Income",
    "Q5_Exercise_Frequency","Q6_Workout_Location","Q7_Primary_Fitness_Goal",
    "Q8_Biggest_Fitness_Challenge","Q9_Diet_Type","Q10_Dietary_Restrictions",
    "Q11_Meal_Management","Q12_Biggest_Diet_Challenge","Q13_Travel_Frequency",
    "Q14_Daily_Routine","Q15_Time_for_Fitness","Q16_Current_Fitness_App_Usage",
    "Q17_Preferred_Features","Q18_Willingness_to_Pay","Q19_What_Would_Make_You_Pay",
    "Q21_Current_Weight_Category","Q22_Problem_Intensity_Score",
    "Q23_Days_Followed_Plan_Last_30_Days","Q24_When_Stop_Routine",
    "Q25_Why_Stopped_Fitness_Apps","Q26_Guaranteed_Results_WTP",
    "Q27_Main_Motivation","Q28_Meal_Control_Level"
]

def split_multi(series):
    return series.fillna("").astype(str).apply(
        lambda x: [item.strip() for item in x.split("|") if item.strip()]
    )

@st.cache_data
def load_data():
    last_error = None
    for path in DATA_FILE_CANDIDATES:
        try:
            df = pd.read_csv(path)
            df = normalize_columns(df)
            df = ensure_required_columns(df)
            return df
        except Exception as exc:
            last_error = exc
    raise FileNotFoundError(f"Could not load any dataset from {DATA_FILE_CANDIDATES}. Last error: {last_error}")

def expand_multiselect(df):
    out = df.copy()
    for col, options in {
        "Q17_Preferred_Features": FEATURE_OPTIONS,
        "Q19_What_Would_Make_You_Pay": PAY_REASON_OPTIONS,
    }.items():
        items = split_multi(out[col] if col in out.columns else pd.Series([""] * len(out)))
        for opt in options:
            out[f"{col}__{opt}"] = items.apply(lambda vals: int(opt in vals))
    return out

def dataset_for_modeling(df):
    data = ensure_required_columns(normalize_columns(df))
    data = expand_multiselect(data)
    data["target_interest"] = data["Q20_Likelihood_to_Use_App"].map(LIKELIHOOD_MAP)
    data["wtp_numeric"] = data["Q18_Willingness_to_Pay"].map(WTP_MAP)
    data["guaranteed_wtp_numeric"] = data["Q26_Guaranteed_Results_WTP"].map(GUARANTEED_WTP_MAP)
    return data

def build_preprocessor(X):
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols

@st.cache_resource
def train_models():
    raw = load_data()
    data = dataset_for_modeling(raw)

    feature_cols = [c for c in data.columns if c not in [
        "Respondent_ID", "Persona_Segment", "Q20_Likelihood_to_Use_App",
        "target_interest", "wtp_numeric", "guaranteed_wtp_numeric"
    ]]
    X = data[feature_cols]
    y = data["target_interest"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    preprocessor, _, _ = build_preprocessor(X)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_leaf=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    clf_results = []
    trained_classifiers = {}
    best_name = None
    best_auc = -1

    for name, model in classifiers.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            probs = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)
            probs = 1 / (1 + np.exp(-scores))
        auc = roc_auc_score(y_test, probs)
        result = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds),
            "ROC AUC": auc,
        }
        clf_results.append(result)
        trained_classifiers[name] = {"pipeline": pipe, "preds": preds, "probs": probs}
        if auc > best_auc:
            best_auc = auc
            best_name = name

    best_clf = trained_classifiers[best_name]["pipeline"]
    best_clf_probs = trained_classifiers[best_name]["probs"]
    best_clf_preds = trained_classifiers[best_name]["preds"]

    # Feature importance for best classifier
    prep = best_clf.named_steps["prep"]
    feature_names = prep.get_feature_names_out()
    model = best_clf.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        coef = model.coef_[0]
        importances = np.abs(coef)
    feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_imp = feat_imp.sort_values("Importance", ascending=False).head(20)

    # Regression
    reg_X = X.copy()
    reg_y = data["guaranteed_wtp_numeric"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(reg_X, reg_y, test_size=0.25, random_state=42)
    reg_preprocessor, _, _ = build_preprocessor(reg_X)
    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=250, max_depth=10, min_samples_leaf=6, random_state=42
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
    }
    reg_results = []
    trained_regressors = {}
    best_reg_name = None
    best_reg_r2 = -10

    for name, model in regressors.items():
        pipe = Pipeline([("prep", reg_preprocessor), ("model", model)])
        pipe.fit(Xr_train, yr_train)
        preds = pipe.predict(Xr_test)
        r2 = r2_score(yr_test, preds)
        reg_results.append({
            "Model": name,
            "MAE": mean_absolute_error(yr_test, preds),
            "RMSE": mean_squared_error(yr_test, preds, squared=False),
            "R2": r2
        })
        trained_regressors[name] = {"pipeline": pipe, "preds": preds}
        if r2 > best_reg_r2:
            best_reg_r2 = r2
            best_reg_name = name

    best_reg = trained_regressors[best_reg_name]["pipeline"]

    # Clustering
    cluster_features = [
        "Q1_Age_Group","Q3_Occupation","Q4_Monthly_Income","Q5_Exercise_Frequency",
        "Q7_Primary_Fitness_Goal","Q10_Dietary_Restrictions","Q11_Meal_Management",
        "Q13_Travel_Frequency","Q14_Daily_Routine","Q16_Current_Fitness_App_Usage",
        "Q18_Willingness_to_Pay","Q21_Current_Weight_Category","Q22_Problem_Intensity_Score",
        "Q23_Days_Followed_Plan_Last_30_Days","Q27_Main_Motivation","Q28_Meal_Control_Level"
    ] + [f"Q17_Preferred_Features__{opt}" for opt in FEATURE_OPTIONS]

    cluster_df = data[cluster_features].copy()
    cprep, _, _ = build_preprocessor(cluster_df)
    Xc = cprep.fit_transform(cluster_df)
    if hasattr(Xc, "toarray"):
        Xc = Xc.toarray()
    scaler = StandardScaler(with_mean=False) if hasattr(Xc, "shape") else StandardScaler()
    Xc_scaled = scaler.fit_transform(Xc)
    kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
    clusters = kmeans.fit_predict(Xc_scaled)
    cluster_labeled = raw.copy()
    cluster_labeled["Cluster"] = clusters

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(Xc_scaled if not hasattr(Xc_scaled, "toarray") else Xc_scaled.toarray())
    cluster_labeled["PCA1"] = pca_2d[:, 0]
    cluster_labeled["PCA2"] = pca_2d[:, 1]

    # Association rules
    assoc_source = expand_multiselect(raw.copy())
    # Build transaction basket from selected categorical columns
    basket_cols = []
    assoc_cols = [
        "Q7_Primary_Fitness_Goal", "Q8_Biggest_Fitness_Challenge", "Q10_Dietary_Restrictions",
        "Q11_Meal_Management", "Q12_Biggest_Diet_Challenge", "Q13_Travel_Frequency",
        "Q24_When_Stop_Routine", "Q25_Why_Stopped_Fitness_Apps", "Q27_Main_Motivation"
    ]
    for col in assoc_cols:
        dummies = pd.get_dummies(assoc_source[col].fillna("Missing"), prefix=col)
        basket_cols.append(dummies)
    for opt in FEATURE_OPTIONS:
        basket_cols.append(assoc_source[[f"Q17_Preferred_Features__{opt}"]].rename(
            columns={f"Q17_Preferred_Features__{opt}": f"Feature_{opt}"}
        ))
    for opt in PAY_REASON_OPTIONS:
        basket_cols.append(assoc_source[[f"Q19_What_Would_Make_You_Pay__{opt}"]].rename(
            columns={f"Q19_What_Would_Make_You_Pay__{opt}": f"PayReason_{opt}"}
        ))
    basket = pd.concat(basket_cols, axis=1).astype(bool)
    frequent_itemsets = apriori(basket, min_support=0.08, use_colnames=True)
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.35)
        rules = rules.sort_values(["lift", "confidence"], ascending=False)
    else:
        rules = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    return {
        "raw": raw,
        "data": data,
        "X_test": X_test,
        "y_test": y_test,
        "class_results": pd.DataFrame(clf_results).sort_values("ROC AUC", ascending=False),
        "trained_classifiers": trained_classifiers,
        "best_classifier_name": best_name,
        "best_classifier": best_clf,
        "best_classifier_preds": best_clf_preds,
        "best_classifier_probs": best_clf_probs,
        "feature_importance": feat_imp,
        "reg_results": pd.DataFrame(reg_results).sort_values("R2", ascending=False),
        "best_regressor_name": best_reg_name,
        "best_regressor": best_reg,
        "clustered": cluster_labeled,
        "association_rules": rules,
        "cluster_model": kmeans,
        "cluster_preprocessor": cprep,
        "cluster_scaler": scaler,
        "cluster_features": cluster_features,
        "feature_cols": feature_cols,
    }

def describe_clusters(clustered):
    summary = clustered.groupby("Cluster").agg(
        respondents=("Respondent_ID", "count"),
        avg_problem_intensity=("Q22_Problem_Intensity_Score", "mean"),
        share_likely=("Q20_Likelihood_to_Use_App", lambda x: np.mean(x.isin(["Likely", "Very likely"]))),
    ).reset_index()
    top_labels = []
    for cluster_id, g in clustered.groupby("Cluster"):
        top_goal = g["Q7_Primary_Fitness_Goal"].mode(dropna=True)
        top_occ = g["Q3_Occupation"].mode(dropna=True)
        top_feature = split_multi(g["Q17_Preferred_Features"].fillna("")).explode().mode()
        label = f"{top_occ.iloc[0] if not top_occ.empty else 'Mixed'} | {top_goal.iloc[0] if not top_goal.empty else 'Mixed'} | {top_feature.iloc[0] if not top_feature.empty else 'Mixed'}"
        top_labels.append((cluster_id, label))
    label_map = dict(top_labels)
    summary["Cluster_Label"] = summary["Cluster"].map(label_map)
    return summary.sort_values("respondents", ascending=False)

def recommendation_logic(prob, pred_wtp, cluster_label):
    if prob >= 0.75 and pred_wtp >= 900:
        return "Premium", "No discount", "Lead with AI coach + all-in-one personalization"
    if prob >= 0.70 and pred_wtp < 900:
        return "Basic / Student", "10-20% starter discount", "Lead with affordability + structured guidance"
    if prob >= 0.50:
        return "Standard", "Free trial", "Lead with convenience + pain-point resolution"
    return "Awareness", "Content-first nurturing", "Lead with education, trust, and before/after value stories"

def prepare_new_data(uploaded_df):
    df = uploaded_df.copy()
    expected = [c for c in BASE_FEATURES if c not in ["Q20_Likelihood_to_Use_App"]]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    # Remove target if user included it
    if "Q20_Likelihood_to_Use_App" in df.columns:
        df = df.drop(columns=["Q20_Likelihood_to_Use_App"])
    if "Respondent_ID" not in df.columns:
        df.insert(0, "Respondent_ID", [f"NEW_{i+1:04d}" for i in range(len(df))])
    ordered = ["Respondent_ID"] + expected
    return df[ordered]

def score_new_customers(new_df, bundle):
    prepared = prepare_new_data(new_df)
    model_input = prepared.drop(columns=["Respondent_ID"]).copy()
    model_ready = expand_multiselect(model_input)
    best_clf = bundle["best_classifier"]
    best_reg = bundle["best_regressor"]
    cluster_df = model_ready[bundle["cluster_features"]].copy()
    Xc = bundle["cluster_preprocessor"].transform(cluster_df)
    if hasattr(Xc, "toarray"):
        Xc = Xc.toarray()
    Xc = bundle["cluster_scaler"].transform(Xc)
    cluster_ids = bundle["cluster_model"].predict(Xc)

    probs = best_clf.predict_proba(model_ready[bundle["feature_cols"]])[:, 1]
    pred_interest = np.where(probs >= 0.5, "Interested", "Not Interested")
    pred_wtp = best_reg.predict(model_ready[bundle["feature_cols"]])
    pred_wtp = np.clip(pred_wtp, 0, None)

    cluster_summary = describe_clusters(bundle["clustered"])
    cluster_label_map = dict(zip(cluster_summary["Cluster"], cluster_summary["Cluster_Label"]))
    cluster_labels = [cluster_label_map.get(c, f"Cluster {c}") for c in cluster_ids]

    recs = [recommendation_logic(p, w, cl) for p, w, cl in zip(probs, pred_wtp, cluster_labels)]
    scored = prepared.copy()
    scored["Predicted_Interest_Class"] = pred_interest
    scored["Predicted_Interest_Probability"] = np.round(probs, 4)
    scored["Predicted_Cluster"] = cluster_labels
    scored["Predicted_Guaranteed_WTP_INR"] = np.round(pred_wtp, 0)
    scored["Recommended_Package"] = [r[0] for r in recs]
    scored["Recommended_Discount"] = [r[1] for r in recs]
    scored["Recommended_Marketing_Message"] = [r[2] for r in recs]
    return scored

bundle = train_models()
raw = bundle["raw"]
cluster_summary = describe_clusters(bundle["clustered"])

st.title("FitFounder Intelligence Hub")
st.caption("Data-driven market intelligence for a personalized fitness and diet app.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Descriptive", "2. Diagnostic", "3. Predictive", "4. Segmentation",
    "5. Association + Prescriptive", "6. Upload & Score New Customers"
])

with tab1:
    st.subheader("Descriptive analytics: what is happening in the market?")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", f"{len(raw):,}")
    c2.metric("Likely/Very likely", f"{(raw['Q20_Likelihood_to_Use_App'].isin(['Likely','Very likely']).mean()*100):.1f}%")
    c3.metric("Current app users", f"{(raw['Q16_Current_Fitness_App_Usage'].eq('Yes').mean()*100):.1f}%")
    c4.metric("Frequent travelers", f"{(raw['Q13_Travel_Frequency'].eq('Frequently').mean()*100):.1f}%")

    colA, colB = st.columns(2)
    with colA:
        age_interest = pd.crosstab(raw["Q1_Age_Group"], raw["Q20_Likelihood_to_Use_App"])
        if PLOTLY_AVAILABLE:
            fig = px.histogram(raw, x="Q1_Age_Group", color="Q20_Likelihood_to_Use_App", barmode="group",
                               category_orders={"Q1_Age_Group": ORDER_MAPS["Q1_Age_Group"] if "Q1_Age_Group" in ORDER_MAPS else None})
            fig.update_layout(height=420, title="Age group vs app interest")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**Age group vs app interest**")
            st.bar_chart(age_interest)
    with colB:
        feature_counts = split_multi(raw["Q17_Preferred_Features"]).explode().value_counts().reset_index()
        feature_counts.columns = ["Feature", "Count"]
        if PLOTLY_AVAILABLE:
            fig = px.bar(feature_counts, x="Count", y="Feature", orientation="h", title="Most requested features")
            fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**Most requested features**")
            st.bar_chart(feature_counts.set_index("Feature"))

    colC, colD = st.columns(2)
    with colC:
        pay_occ = pd.crosstab(raw["Q18_Willingness_to_Pay"], raw["Q3_Occupation"])
        if PLOTLY_AVAILABLE:
            fig = px.histogram(raw, x="Q18_Willingness_to_Pay", color="Q3_Occupation", barmode="group",
                               title="Monthly willingness to pay by occupation")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**Monthly willingness to pay by occupation**")
            st.bar_chart(pay_occ)
    with colD:
        challenge_mix = pd.crosstab(raw["Q8_Biggest_Fitness_Challenge"], raw["Q12_Biggest_Diet_Challenge"])
        if PLOTLY_AVAILABLE:
            fig = px.histogram(raw, x="Q8_Biggest_Fitness_Challenge", color="Q12_Biggest_Diet_Challenge",
                               title="Top fitness and diet pain points")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**Top fitness and diet pain points**")
            st.bar_chart(challenge_mix)

with tab2:
    st.subheader("Diagnostic analytics: why are people interested or not interested?")
    col1, col2 = st.columns(2)
    with col1:
        pivot = pd.crosstab(raw["Q13_Travel_Frequency"], raw["Q20_Likelihood_to_Use_App"], normalize="index")
        if PLOTLY_AVAILABLE:
            pivot_melt = pivot.reset_index().melt(id_vars="Q13_Travel_Frequency", var_name="Interest", value_name="Share")
            fig = px.bar(pivot_melt, x="Q13_Travel_Frequency", y="Share", color="Interest", barmode="stack",
                         title="Travel frequency vs interest")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**Travel frequency vs interest**")
            st.bar_chart(pivot)
    with col2:
        app_users = raw[raw["Q16_Current_Fitness_App_Usage"] == "Yes"].copy()
        churn_order = app_users["Q25_Why_Stopped_Fitness_Apps"].value_counts().reset_index()
        churn_order.columns = ["Reason", "Count"]
        if PLOTLY_AVAILABLE:
            fig = px.bar(churn_order, x="Reason", y="Count", title="Why current app users churn")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**Why current app users churn**")
            st.bar_chart(churn_order.set_index("Reason"))

    st.markdown("**Best explanatory signals from the top classifier**")
    if PLOTLY_AVAILABLE:
        fig = px.bar(bundle["feature_importance"].sort_values("Importance"),
                     x="Importance", y="Feature", orientation="h", title="Top feature importance")
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(bundle["feature_importance"].set_index("Feature"))

with tab3:
    st.subheader("Predictive analytics: who is likely to convert, and what might they pay?")
    st.markdown(f"**Best classification model:** {bundle['best_classifier_name']}")
    st.dataframe(bundle["class_results"].style.format({
        "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
        "F1-Score": "{:.3f}", "ROC AUC": "{:.3f}"
    }), use_container_width=True)

    y_test = bundle["y_test"]
    probs = bundle["best_classifier_probs"]
    preds = bundle["best_classifier_preds"]
    fpr, tpr, _ = roc_curve(y_test, probs)

    col1, col2 = st.columns(2)
    with col1:
        if PLOTLY_AVAILABLE:
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
            roc_fig.update_layout(title="ROC curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=420)
            st.plotly_chart(roc_fig, use_container_width=True)
        else:
            st.markdown("**ROC curve data**")
            st.line_chart(pd.DataFrame({"ROC Curve": tpr, "Baseline": fpr}, index=fpr))
    with col2:
        cm = confusion_matrix(y_test, preds)
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        if PLOTLY_AVAILABLE:
            fig = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion matrix")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**Confusion matrix**")
            st.dataframe(cm_df, use_container_width=True)

    st.markdown(f"**Best regression model:** {bundle['best_regressor_name']} (target: guaranteed-results WTP)")
    st.dataframe(bundle["reg_results"].style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}), use_container_width=True)

with tab4:
    st.subheader("Customer segmentation: who should we target first?")
    if PLOTLY_AVAILABLE:
        fig = px.scatter(bundle["clustered"], x="PCA1", y="PCA2", color=bundle["clustered"]["Cluster"].astype(str),
                         hover_data=["Q3_Occupation", "Q7_Primary_Fitness_Goal", "Q18_Willingness_to_Pay"],
                         title="Cluster map (PCA projection)")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("**Cluster map (PCA projection)**")
        st.scatter_chart(bundle["clustered"], x="PCA1", y="PCA2", color="Cluster")

    st.dataframe(cluster_summary.style.format({
        "avg_problem_intensity": "{:.2f}",
        "share_likely": "{:.1%}"
    }), use_container_width=True)

with tab5:
    st.subheader("Association rules + prescriptive actions")
    rules = bundle["association_rules"].copy()
    if not rules.empty:
        show_rules = rules.head(20).copy()
        show_rules["antecedents"] = show_rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
        show_rules["consequents"] = show_rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
        st.dataframe(show_rules[["antecedents", "consequents", "support", "confidence", "lift"]].style.format({
            "support": "{:.3f}", "confidence": "{:.3f}", "lift": "{:.3f}"
        }), use_container_width=True)
    else:
        st.info("No association rules met the current support/confidence thresholds.")

    st.markdown("### Founder action center")
    cluster_summary_view = cluster_summary.copy()
    cluster_summary_view["Priority"] = np.where(
        (cluster_summary_view["share_likely"] >= 0.60), "Target first", "Nurture / test"
    )
    cluster_summary_view["Suggested Offer"] = np.select(
        [
            cluster_summary_view["Cluster_Label"].str.contains("Student", na=False),
            cluster_summary_view["Cluster_Label"].str.contains("Working Professional|Business Owner", na=False),
        ],
        [
            "Student / entry bundle with starter discount",
            "Premium all-in-one bundle with convenience messaging",
        ],
        default="Core guided plan with free trial"
    )
    st.dataframe(cluster_summary_view[["Cluster", "Cluster_Label", "respondents", "share_likely", "Priority", "Suggested Offer"]]
                 .style.format({"share_likely": "{:.1%}"}), use_container_width=True)

with tab6:
    st.subheader("Upload future customer data and score new prospects")
    st.markdown(
        "Upload a CSV with the same survey columns. The target column `Q20_Likelihood_to_Use_App` is optional for uploads. "
        "The app will predict inclination, estimated willingness to pay, cluster assignment, and a recommended marketing strategy."
    )

    with st.expander("Expected columns"):
        st.write(["Respondent_ID"] + [c for c in BASE_FEATURES if c != "Q20_Likelihood_to_Use_App"])

    uploaded = st.file_uploader("Upload new customer CSV", type=["csv"])
    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        scored = score_new_customers(new_df, bundle)
        st.success("Scoring complete.")
        st.dataframe(scored.head(20), use_container_width=True)

        out = io.BytesIO()
        scored.to_csv(out, index=False)
        st.download_button(
            label="Download scored customers CSV",
            data=out.getvalue(),
            file_name="scored_new_customers.csv",
            mime="text/csv"
        )

        st.markdown("### Lead distribution")
        if PLOTLY_AVAILABLE:
            fig = px.histogram(scored, x="Predicted_Interest_Class", color="Recommended_Package",
                               title="Predicted lead mix by recommended package")
            st.plotly_chart(fig, use_container_width=True)
        else:
            lead_mix = pd.crosstab(scored["Predicted_Interest_Class"], scored["Recommended_Package"])
            st.bar_chart(lead_mix)
    else:
        st.info("Use the included `sample_new_customers.csv` file to test the upload flow.")
