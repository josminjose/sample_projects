# HackerRank Challenge Recommendation — End-to-End ML (Classification + Ranking)

**Goal**: Given a (user, challenge) pair, predict **probability of solve** and rank challenges per user.
This mirrors the *HackerRank Challenge Recommendation* idea; data here is synthetic but schema-aligned for interviews.

## Features
- **Binary classification**: will the user solve a recommended challenge?
- **Ranking**: evaluate MAP@K / Recall@K per user.
- **Threshold tuning**: precision–recall trade-off, confusion matrix, PR/ROC curves.
- **Models**: Logistic Regression & Gradient Boosting (scikit-learn).
- **Artifacts**: saved models + JSON metrics.

## Quick Start
```bash
pip install -r requirements.txt

# 1) Generate synthetic interactions + train & evaluate
python app/train_eval.py

# 2) Launch interactive dashboard
streamlit run app/streamlit_app.py
```
Open Streamlit to explore **threshold** → precision/recall, **PR/ROC**, **confusion matrix**, and **top-K ranking** metrics.

## Data (synthetic)
- `users.csv`: user-level activity features
- `challenges.csv`: challenge-level difficulty & tags
- `interactions.csv`: (user, challenge, solved)
- `pairs.csv`: training pairs with engineered features

## Talking Points
- Why **precision vs recall** matters (false positives waste review slots; false negatives miss good recommendations).
- Ranking metrics (**MAP@K, Recall@K**) complement thresholded classification.
- Cold-start mitigation ideas (content features/tags), calibration, cost-sensitive thresholds.

---
*This project is inspired by the HackerRank "Challenge Recommendation" problem; data provided here is synthetic for demonstration.*
