"""
Lab 06: Scikit-learn Pipeline Value Forecasting

Uses scikit-learn to train a simple deal close probability model
based on CRM features, demonstrating ML fundamentals without PyTorch/TF overhead.

Features used:
  - days_until_close
  - deal_stage (encoded)
  - amount
  - lead_score

Target: Will this deal close?

INTERVIEW TALKING POINT:
  "Before adding the AI agent, I explored classical ML for deal scoring.
   scikit-learn's Pipeline class handles preprocessing + model in one object,
   preventing data leakage by fitting transformers only on training data."

Requirements: pip install scikit-learn numpy pandas
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

# ── Synthetic CRM data ────────────────────────────────────────────────────────
np.random.seed(42)
N = 500

STAGES = ["prospect", "qualified", "demo_scheduled", "demo_completed",
          "proposal_sent", "negotiation"]
STAGE_WEIGHTS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]  # base close probability by stage

def generate_deal_data(n=N):
    stage_idx = np.random.randint(0, len(STAGES), n)
    base_prob = np.array([STAGE_WEIGHTS[i] for i in stage_idx])

    data = {
        "stage": [STAGES[i] for i in stage_idx],
        "amount": np.random.lognormal(mean=10, sigma=1.5, size=n),
        "days_until_close": np.random.randint(1, 180, n),
        "lead_score": np.random.randint(20, 100, n),
        "num_activities": np.random.randint(1, 30, n),
    }

    # Target: closed_won = 1 (influenced by stage, lead score, activities)
    noise = np.random.normal(0, 0.15, n)
    prob = np.clip(
        base_prob
        + 0.002 * (data["lead_score"] - 60)
        + 0.005 * (data["num_activities"] - 10)
        + noise,
        0.01, 0.99
    )
    data["closed_won"] = (np.random.random(n) < prob).astype(int)
    return pd.DataFrame(data)

df = generate_deal_data()
print(f"Dataset: {len(df)} deals | Close rate: {df['closed_won'].mean():.1%}")

# ── Feature Engineering ───────────────────────────────────────────────────────
X = df.drop("closed_won", axis=1)
y = df["closed_won"]

# WHY ColumnTransformer:
#   Applies different preprocessing to different column types simultaneously.
#   StandardScaler for numerics (mean=0, std=1).
#   OrdinalEncoder for stage (ordinal: prospect < qualified < ... < negotiation).
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), ["amount", "days_until_close", "lead_score", "num_activities"]),
    ("cat", OrdinalEncoder(categories=[STAGES]), ["stage"]),
])

# WHY Pipeline:
#   Chains preprocessing + model into one object.
#   Prevents data leakage: scaler is fit ONLY on training data,
#   then applied to test data — not fit on the full dataset first.
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )),
])

# ── Train / Evaluate ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# ── Cross Validation ──────────────────────────────────────────────────────────
# WHY cross_val_score:
#   A single train/test split can be lucky or unlucky.
#   5-fold CV trains on 5 different splits and averages the score — more reliable.
cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print(f"\n5-fold CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── Feature Importance ────────────────────────────────────────────────────────
feature_names = ["amount", "days_until_close", "lead_score", "num_activities", "stage"]
importances = model.named_steps["classifier"].feature_importances_
print("\nFeature Importances:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {name:<20} {imp:.3f} {bar}")

# ── Predict on a new deal ─────────────────────────────────────────────────────
new_deal = pd.DataFrame([{
    "stage": "proposal_sent",
    "amount": 150_000,
    "days_until_close": 30,
    "lead_score": 85,
    "num_activities": 12,
}])

prob = model.predict_proba(new_deal)[0][1]
print(f"\n\nPrediction for new deal (proposal_sent, $150K, lead_score=85):")
print(f"  Close probability: {prob:.1%}")
print("\nDone — scikit-learn Pipeline model complete.")
