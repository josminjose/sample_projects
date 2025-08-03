# app/streamlit_app.py
import os, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="HackerRank Challenge Recommendation — ML", layout="wide")
st.title("HackerRank Challenge Recommendation — Precision/Recall & Ranking Demo")

# -------- Path anchoring (robust) --------
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE    = ROOT / "data" / "pairs.csv"
REPORTS_DIR  = ROOT / "reports"
REPORTS_FILE = REPORTS_DIR / "metrics.json"     # <-- file to open
MODELS_DIR   = ROOT / "models"

@st.cache_data
def load_pairs():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_reports():
    if not REPORTS_FILE.exists():
        st.error(f"metrics.json not found at:\n{REPORTS_FILE}\n\n"
                 "Run `python app\\train_eval.py` from the project root to generate it.")
        st.stop()
    with REPORTS_FILE.open("r") as f:
        return json.load(f)

# ---------- Load data & metrics ----------
pairs = load_pairs()
reports = load_reports()

model_name = st.selectbox("Model", list(reports.keys()), index=0)
metrics = reports[model_name]

st.subheader("Test Metrics @ threshold = 0.5")
st.json(metrics["test"]["classification_report@0.5"])
st.write({k:v for k,v in metrics["test"].items() if k in ["average_precision","roc_auc","map@5","recall@5"]})

# ---------- PR/ROC & Threshold ----------
st.divider()
st.subheader("Precision–Recall Curve & Threshold Tuning")

pr  = np.array(metrics["curves"]["precision"])
rc  = np.array(metrics["curves"]["recall"])
thr = np.array(metrics["curves"]["thresholds"])

th = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)
idx = (np.abs(thr - th)).argmin() if len(thr) > 0 else 0
cur_p = float(pr[idx]) if len(pr)  > idx else float("nan")
cur_r = float(rc[idx]) if len(rc)  > idx else float("nan")

col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(rc, pr)
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision–Recall Curve")
    if len(thr) > 0:
        ax1.scatter(rc[idx], pr[idx])
        ax1.text(rc[idx], pr[idx], f"  t≈{thr[idx]:.2f}\n  P={cur_p:.2f}, R={cur_r:.2f}")
    st.pyplot(fig1)

with col2:
    fpr = np.array(metrics["curves"]["fpr"])
    tpr = np.array(metrics["curves"]["tpr"])
    roc_auc = float(metrics["test"]["roc_auc"])
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    ax2.plot([0,1],[0,1], linestyle="--", color="gray")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)

st.markdown(f"**Selected threshold:** {th:.2f} → approx **Precision {cur_p:.2f}**, **Recall {cur_r:.2f}**")

# ---------- Confusion Matrix at chosen threshold ----------
y = pairs["solved"].values
X = pairs.drop(columns=["solved","user_id","challenge_id"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)

model_path = MODELS_DIR / f"{model_name}.joblib"
pipe = joblib.load(model_path)
y_proba = pipe.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= th).astype(int)
cm = confusion_matrix(y_test, y_pred)

st.subheader("Confusion Matrix (Test) at Selected Threshold")
fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax3)
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
st.pyplot(fig3)

st.subheader("Classification Report at Selected Threshold")
report = classification_report(y_test, y_pred, output_dict=True)
st.json(report)

# ---------- User-level Top-5 preview ----------
st.divider()
st.subheader("User-level Ranking Preview (Top-5)")
pairs_idx = np.arange(len(pairs))
_, test_idx = train_test_split(pairs_idx, test_size=0.2, random_state=17, stratify=y)
test_pairs = pairs.iloc[test_idx].copy()
test_pairs["proba"] = y_proba
uid = st.selectbox("Pick a user id from the test set", sorted(test_pairs["user_id"].unique())[:50])
uview = test_pairs[test_pairs["user_id"] == uid].sort_values("proba", ascending=False).head(5)
st.dataframe(uview[["user_id","challenge_id","proba","solved"]])
