# app/train_eval.py
from __future__ import annotations
from pathlib import Path
import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score,
    classification_report, confusion_matrix
)

RANDOM_STATE = 17

# -------- Path anchoring (robust) --------
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE    = ROOT / "data" / "pairs.csv"
REPORTS_DIR  = ROOT / "reports"
MODELS_DIR   = ROOT / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_FILE)
    y = df["solved"].values
    X = df.drop(columns=["solved","user_id","challenge_id"])
    meta = df[["user_id","challenge_id"]]
    return X, y, meta

def fit_and_eval(model, name):
    X, y, meta = load_data()
    X_tr, X_te, y_tr, y_te, meta_tr, meta_te = train_test_split(
        X, y, meta, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    pipe.fit(X_tr, y_tr)

    y_proba = pipe.predict_proba(X_te)[:, 1]
    y_pred05 = (y_proba >= 0.5).astype(int)

    pr, rc, thr = precision_recall_curve(y_te, y_proba)
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    ap = average_precision_score(y_te, y_proba)
    roc = roc_auc_score(y_te, y_proba)
    report = classification_report(y_te, y_pred05, output_dict=True)
    cm = confusion_matrix(y_te, y_pred05).tolist()

    # Simple per-user ranking metrics (MAP@5 approx, Recall@5)
    te_df = meta_te.copy()
    te_df["proba"] = y_proba
    te_df["label"] = y_te
    K = 5
    map_k_list, rec_k_list = [], []
    for uid, grp in te_df.groupby("user_id"):
        grp_sorted = grp.sort_values("proba", ascending=False)
        topk = grp_sorted.head(K)
        pos_positions = [i+1 for i, r in enumerate(topk.itertuples()) if r.label==1]
        ap_k = 1.0 / pos_positions[0] if pos_positions else 0.0
        map_k_list.append(ap_k)
        rec_k_list.append(topk["label"].sum() / max(1, grp["label"].sum()))
    map_k = float(np.mean(map_k_list))
    recall_k = float(np.mean(rec_k_list))

    out = {
        "name": name,
        "test": {
            "average_precision": float(ap),
            "roc_auc": float(roc),
            "confusion_matrix@0.5": cm,
            "classification_report@0.5": report,
            "map@5": map_k,
            "recall@5": recall_k
        },
        "curves": {
            "precision": pr.tolist(),
            "recall": rc.tolist(),
            "thresholds": thr.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }
    }

    joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")
    return out

def main():
    results = {}
    results["logreg"]  = fit_and_eval(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), "logreg")
    results["gboost"]  = fit_and_eval(GradientBoostingClassifier(random_state=RANDOM_STATE), "gboost")

    with (REPORTS_DIR / "metrics.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved models to: {MODELS_DIR}")
    print(f"Saved metrics to: {REPORTS_DIR / 'metrics.json'}")

if __name__ == "__main__":
    main()
