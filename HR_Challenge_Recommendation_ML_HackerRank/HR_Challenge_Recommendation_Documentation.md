
# HackerRank Challenge Recommendation — End‑to‑End ML Documentation

**Goal:** For a given *(user, challenge)* pair, predict the **probability of solve** and rank challenges per user.  
Inspired by the real *HackerRank Challenge Recommendation* problem; dataset is synthetic but schema‑aligned.

## 1) System Architecture
![Architecture Diagram](HR_Challenge_Recommendation_Architecture.png)

**Components**
1. **Data Sources (`data/`)** — users, challenges, interactions
2. **Feature Builder** — joins & engineers features into `pairs.csv`
3. **Training & Evaluation** — Logistic Regression & Gradient Boosting, PR/ROC, MAP@5, Recall@5
4. **Dashboard** — Streamlit app for threshold tuning, metrics, Top‑K preview
5. **Storage** — data, models, reports

## 2) Data & Features
- `users.csv`: skill, activity, preferences
- `challenges.csv`: difficulty, tags, popularity
- `interactions.csv`: solved flag
- `pairs.csv`: merged features incl. `align`, normalized counts, skill‑difficulty gap

## 3) Modeling
- Preprocessing: StandardScaler
- Models: LogisticRegression, GradientBoostingClassifier
- Metrics: Average Precision, ROC AUC, Confusion Matrix, MAP@5, Recall@5

## 4) Evaluation
- **Classification:** Precision, Recall, PR curve, ROC
- **Ranking:** MAP@5, Recall@5

## 5) Dashboard
- Model selector
- Threshold slider → PR trade‑off
- Confusion matrix
- Top‑5 recommendations/user

## 6) Run Instructions
```bash
pip install -r requirements.txt
python app/train_eval.py
streamlit run app/streamlit_app.py
```

## 7) Extensibility
- Add advanced models (XGBoost, LightGBM)
- Calibration
- Cold‑start solutions
- Production deployment via API
