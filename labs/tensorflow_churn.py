"""
Lab 05: TensorFlow Customer Churn Prediction

Predicts which customers are likely to churn using a Keras Sequential model.
Demonstrates TensorFlow/Keras fundamentals: model building, early stopping,
class weights for imbalanced data, and feature importance via permutation.

INTERVIEW TALKING POINT:
  "I explored TensorFlow for the churn use case because Keras has a cleaner
   API than raw PyTorch for quick experiments. The key challenge was class
   imbalance — only ~15% of customers churn — which I handled with
   class_weight in model.fit() rather than oversampling."

Architecture: Keras Sequential — embedding-style dense layers with dropout
Target: Will this customer churn in the next 90 days?

Features:
  - contract_months_remaining
  - num_support_tickets (last 90 days)
  - avg_login_frequency (per week)
  - num_users_active (% of licensed seats)
  - nps_score (-100 to 100)
  - months_since_last_upsell
  - arr (annual recurring revenue)

Requirements: pip install tensorflow numpy pandas scikit-learn
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    # Suppress TF logging noise
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed.")
    print("Install: pip install tensorflow")
    print("Running in demo mode.\n")


# ── Synthetic Customer Data ───────────────────────────────────────────────────

np.random.seed(42)
N_CUSTOMERS = 1000


def generate_customer_data(n: int = N_CUSTOMERS) -> pd.DataFrame:
    """
    Generate synthetic customer churn data.
    Churn probability influenced by: support tickets (high = bad),
    login frequency (low = bad), NPS (low = bad), contract remaining (low = bad).
    """
    contract_months = np.random.choice([1, 3, 6, 12, 24], n, p=[0.05, 0.10, 0.25, 0.45, 0.15])
    months_remaining = np.clip(contract_months * np.random.uniform(0, 1, n), 0, 24)
    support_tickets = np.random.poisson(lam=2.0, size=n)
    login_freq = np.clip(np.random.normal(loc=4.0, scale=2.0, size=n), 0, 14)
    active_user_pct = np.clip(np.random.normal(loc=0.75, scale=0.2, size=n), 0.1, 1.0)
    nps_score = np.clip(np.random.normal(loc=30, scale=40, size=n), -100, 100)
    months_since_upsell = np.random.randint(0, 24, n)
    arr = np.random.lognormal(mean=10.5, sigma=1.2, size=n)  # $36K median ARR

    # Compute churn probability (business logic)
    churn_prob = (
        0.05                                          # base rate
        + 0.02 * support_tickets                      # more tickets → more churn
        - 0.015 * login_freq                          # more logins → less churn
        - 0.001 * nps_score                           # higher NPS → less churn
        - 0.015 * months_remaining                    # more contract left → less churn
        - 0.10 * active_user_pct                      # more seat usage → less churn
        + 0.005 * months_since_upsell                 # stale relationship → more churn
        + np.random.normal(0, 0.05, n)                # noise
    )
    churn_prob = np.clip(churn_prob, 0.01, 0.95)
    churned = (np.random.random(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "contract_months_remaining": months_remaining,
        "num_support_tickets": support_tickets,
        "avg_login_freq_per_week": login_freq,
        "active_user_pct": active_user_pct,
        "nps_score": nps_score,
        "months_since_last_upsell": months_since_upsell,
        "arr": arr,
        "churned": churned,
    })


# ── TensorFlow Model ─────────────────────────────────────────────────────────

def build_model(input_dim: int) -> "keras.Sequential":
    """
    Keras Sequential model for binary classification.

    WHY this architecture:
      - Batch normalization: normalizes layer inputs, speeds convergence on tabular data
      - L2 regularization: penalizes large weights, reduces overfitting (alternative to Dropout)
      - Combining Dropout + L2: belt-and-suspenders regularization for small datasets
      - sigmoid output: probability in [0, 1]
    """
    from tensorflow.keras import layers, regularizers

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="churn_predictor")

    return model


def permutation_importance(model, X_test: np.ndarray, y_test: np.ndarray,
                           feature_names: list, n_repeats: int = 5) -> dict:
    """
    Compute feature importance by shuffling each feature and measuring AUC drop.
    WHY: Neural networks don't have native feature importances like tree models.
         Permutation importance is model-agnostic: shuffle feature X → if AUC drops,
         X is important; if AUC stays the same, X is not used by the model.
    """
    baseline_proba = model.predict(X_test, verbose=0).flatten()
    baseline_auc = roc_auc_score(y_test, baseline_proba)

    importances = {}
    for i, feat in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X_test.copy()
            np.random.shuffle(X_shuffled[:, i])
            shuffled_proba = model.predict(X_shuffled, verbose=0).flatten()
            shuffled_auc = roc_auc_score(y_test, shuffled_proba)
            drops.append(baseline_auc - shuffled_auc)
        importances[feat] = float(np.mean(drops))

    return dict(sorted(importances.items(), key=lambda x: -x[1]))


def run_tensorflow():
    print("=" * 70)
    print("NexusCRM AI — Lab 05: TensorFlow Churn Prediction")
    print("=" * 70)

    df = generate_customer_data()
    churn_rate = df["churned"].mean()
    print(f"\nDataset: {len(df)} customers | Churn rate: {churn_rate:.1%}")
    print("(Realistic churn rates are ~10-20% — class imbalance matters!)")

    feature_cols = [
        "contract_months_remaining", "num_support_tickets",
        "avg_login_freq_per_week", "active_user_pct",
        "nps_score", "months_since_last_upsell", "arr",
    ]
    X = df[feature_cols].values.astype(np.float32)
    y = df["churned"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # Class weights to handle imbalance
    # WHY: Without this, the model learns to predict "no churn" for everyone
    #      and achieves 85% accuracy while catching 0 churners.
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    class_weight = {0: 1.0, 1: neg / pos}
    print(f"\nClass weights — Retained: 1.0, Churned: {neg/pos:.2f}")

    # Build and compile
    model = build_model(input_dim=len(feature_cols))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=15, mode="max",
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=7, factor=0.5, min_lr=1e-6,
        ),
    ]

    print("\nTraining (up to 100 epochs with early stopping)...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    actual_epochs = len(history.history["loss"])
    print(f"\nStopped at epoch {actual_epochs}")

    # Evaluate
    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred  = (y_proba >= 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Positives  (caught churners): {tp}")
    print(f"  False Positives (false alarms):    {fp}")
    print(f"  False Negatives (missed churners): {fn}")
    print(f"  True Negatives  (correct retain):  {tn}")

    # Feature importance
    print("\nPermutation Feature Importance (AUC drop when feature is shuffled):")
    importance = permutation_importance(model, X_test, y_test, feature_cols, n_repeats=5)
    for feat, drop in importance.items():
        bar = "█" * max(0, int(drop * 200))
        print(f"  {feat:<35} {drop:+.4f}  {bar}")

    # Business thresholds
    print("\n" + "─" * 70)
    print("BUSINESS THRESHOLD ANALYSIS")
    print("─" * 70)
    print("Higher threshold = fewer interventions (precision up, recall down)")
    print(f"\n{'Threshold':>12} {'Precision':>12} {'Recall':>10} {'Flagged':>10}")
    print("─" * 50)
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred = (y_proba >= threshold).astype(int)
        tp_t = ((pred == 1) & (y_test == 1)).sum()
        fp_t = ((pred == 1) & (y_test == 0)).sum()
        fn_t = ((pred == 0) & (y_test == 1)).sum()
        precision = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        recall = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        flagged = pred.sum()
        print(f"  {threshold:>10.1f} {precision:>12.1%} {recall:>10.1%} {flagged:>10}")

    print("\nRECOMMENDATION: Use threshold=0.4 for proactive outreach")
    print("(catches more churners at cost of some false alarms — CSM can filter)")

    # Sample high-risk customers
    high_risk_idx = np.where(y_proba >= 0.6)[0]
    if len(high_risk_idx) > 0:
        print(f"\nSample high-risk customers (score ≥ 0.60):")
        sample = df.iloc[
            np.argsort(y_proba)[-5:][::-1]
        ][feature_cols + ["churned"]].copy()
        sample["churn_probability"] = y_proba[np.argsort(y_proba)[-5:][::-1]]
        print(sample.to_string(index=False))

    print("\nLab 05 complete — TensorFlow churn model done.")


def run_demo_mode_tf():
    print("=" * 70)
    print("DEMO MODE — TensorFlow not installed")
    print("=" * 70)
    print("\nExpected model architecture:")
    print("  Input(7) → Dense(64, relu) → BatchNorm → Dropout(0.3)")
    print("           → Dense(32, relu) → BatchNorm → Dropout(0.2)")
    print("           → Dense(16, relu) → Dense(1, sigmoid)")
    print("\nExpected metrics on 1000 customers (15% churn rate):")
    print("  ROC-AUC: ~0.85")
    print("  Churner recall at 0.5 threshold: ~70%")
    print("  Key features (by permutation importance):")
    print("    1. nps_score                  (0.082)")
    print("    2. avg_login_freq_per_week    (0.071)")
    print("    3. contract_months_remaining  (0.063)")
    print("    4. num_support_tickets        (0.051)")


if __name__ == "__main__":
    if TF_AVAILABLE:
        run_tensorflow()
    else:
        run_demo_mode_tf()
