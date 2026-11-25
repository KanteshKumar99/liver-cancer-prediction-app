import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

DATA_FILE = "liver_cancer_prediction.csv.gz"
DROP_COLS = ["Prediction", "Country", "Region"]  # same features as project.py

st.set_page_config(page_title="Liver Cancer Prediction", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .main-title {font-size:2.2rem;font-weight:700;color:#1E3A8A;text-align:center;margin-bottom:0}
    .sub-title {text-align:center;color:#666;margin-bottom:22px}
    .metric-card {background:#F8F9FA;color:#111;padding:14px;border-radius:12px;text-align:center;
                  box-shadow:0 1px 4px rgba(0,0,0,.08)}
    .result-high {background:#FDECEC;color:#111;border-left:6px solid #DC2626;padding:18px;border-radius:10px}
    .result-low {background:#E9F9EE;color:#111;border-left:6px solid #16A34A;padding:18px;border-radius:10px}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Model train ho raha hai (pehli baar ~1 min lagta hai)...")
def build_model():
    df = pd.read_csv(DATA_FILE)
    y = df["Prediction"].map({"No": 0, "Yes": 1})
    X = df.drop(columns=DROP_COLS)

    # Label-encode categoricals (alphabetical, same as LabelEncoder in project.py)
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    mappings = {c: {v: i for i, v in enumerate(sorted(X[c].unique()))} for c in cat_cols}
    for c in cat_cols:
        X[c] = X[c].map(mappings[c])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(
        n_estimators=100, max_depth=14, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-Score": f1_score(y_test, pred, zero_division=0),
        "Baseline": float((y_test == 0).mean()),
    }
    num_stats = X[[c for c in X.columns if c not in cat_cols]].describe().loc[["min", "50%", "max"]]
    return model, mappings, list(X.columns), metrics, num_stats, df["Prediction"].value_counts()


model, mappings, feature_order, metrics, num_stats, class_counts = build_model()

st.markdown("<div class='main-title'>🩺 Liver Cancer Prediction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Random Forest classifier trained on the liver cancer dataset</div>",
    unsafe_allow_html=True,
)

cols = st.columns(4)
for col, name in zip(cols, ["Accuracy", "Precision", "Recall", "F1-Score"]):
    val = metrics[name]
    shown = f"{val * 100:.2f}%" if name == "Accuracy" else f"{val:.2f}"
    col.markdown(f"<div class='metric-card'><h3>{shown}</h3>{name}</div>", unsafe_allow_html=True)

st.caption(
    f"Test-set metrics (80/20 split, random_state=42). Precision/Recall/F1 are for the 'Yes' class. "
    f"Majority-class baseline accuracy: {metrics['Baseline'] * 100:.2f}%."
)
if metrics["Accuracy"] - metrics["Baseline"] < 0.01:
    st.warning(
        "Is dataset mein features aur target ke beech bahut kam signal hai, isliye model majority class "
        "('No') ke baseline se behtar nahi hai. Predictions ko sirf demo ke liye samjho."
    )
st.markdown("---")

# ---------------- Sidebar inputs ----------------
st.sidebar.header("📋 Patient / Region Information")
values = {}
for feat in feature_order:
    label = feat.replace("_", " ")
    if feat in mappings:
        values[feat] = st.sidebar.selectbox(label, list(mappings[feat].keys()))
    else:
        lo, mid, hi = (float(num_stats.loc[k, feat]) for k in ["min", "50%", "max"])
        if feat == "Age":
            values[feat] = st.sidebar.slider(label, int(lo), int(hi), int(mid))
        else:
            values[feat] = st.sidebar.number_input(label, lo, hi, mid)

go = st.sidebar.button("🔍 Predict", use_container_width=True)

if go:
    row = {f: (mappings[f][v] if f in mappings else v) for f, v in values.items()}
    X_in = pd.DataFrame([row])[feature_order]
    prob = float(model.predict_proba(X_in)[0][1])
    pred = prob >= 0.5

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Prediction Result")
        if pred:
            st.markdown(
                f"<div class='result-high'><h3>⚠️ Liver Cancer: Likely (Yes)</h3>"
                f"<p>Estimated probability: <b>{prob * 100:.2f}%</b></p>"
                f"<p>Model-based estimate, not a medical diagnosis. Please consult a doctor.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='result-low'><h3>✅ Liver Cancer: Unlikely (No)</h3>"
                f"<p>Estimated probability: <b>{prob * 100:.2f}%</b></p>"
                f"<p>Model-based estimate, not a medical diagnosis.</p></div>",
                unsafe_allow_html=True,
            )
        st.write("**Probability of 'Yes'**")
        st.progress(min(max(prob, 0.0), 1.0))
    with right:
        st.subheader("Input Summary")
        st.dataframe(pd.DataFrame({"Value": [str(v) for v in values.values()]}, index=list(values.keys())))
else:
    st.info("👈 Sidebar mein details fill karo aur **Predict** button dabao.")

st.markdown("---")
with st.expander("📊 Feature importance (Random Forest)"):
    imp = pd.Series(model.feature_importances_, index=feature_order).sort_values(ascending=False)
    st.bar_chart(imp)

with st.expander("📈 Dataset class distribution"):
    st.bar_chart(class_counts)

st.caption("⚠️ Disclaimer: Educational/demo purpose only, medical advice ka substitute nahi.")
