"""
In production, replace generate_synthetic_data() with your
Shopify + Supabase query functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report,
    ConfusionMatrixDisplay, roc_curve
)
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ──────────────────────────────────────────────
# 1. DATA LAYER
#    In production: replace with Shopify / Supabase queries
# ──────────────────────────────────────────────

def generate_synthetic_data(n: int = 5000) -> pd.DataFrame:
    """
    Simulates Riversol customer records after sample kit order.
    Distributions are calibrated to mimic a real DTC skincare cohort.

    Columns match what you'd extract from:
      - Shopify: order date, product pages, revenue intent
      - Supabase: skin profile form submitted at sample checkout
      - Email platform (Klaviyo): open/click rates
    """
    skin_concerns = ["Rosacea", "Anti-aging", "Sensitivity", "Dryness", "Oily/Acne"]
    age_ranges    = ["18-29", "30-39", "40-49", "50-59", "60+"]
    climates      = ["Dry", "Humid", "Temperate", "Cold"]
    skin_types    = ["Dry", "Oily", "Combination", "Normal", "Sensitive"]

    # Base conversion rates by concern — Rosacea patients are most motivated
    conv_rates = {
        "Rosacea":     0.38,
        "Anti-aging":  0.31,
        "Sensitivity": 0.27,
        "Dryness":     0.22,
        "Oily/Acne":   0.18,
    }
    age_bonus = {"18-29": 0.00, "30-39": 0.03, "40-49": 0.08, "50-59": 0.06, "60+": 0.04}

    records = []
    for i in range(n):
        concern   = np.random.choice(skin_concerns)
        age       = np.random.choice(age_ranges, p=[0.15, 0.25, 0.28, 0.20, 0.12])
        climate   = np.random.choice(climates)
        skin_type = np.random.choice(skin_types)

        # Behavioral signals (available ~7 days post-sample dispatch)
        email_open_rate = np.clip(np.random.beta(3, 5), 0.05, 0.85)
        email_ctr       = np.clip(email_open_rate * np.random.uniform(0.1, 0.6), 0.01, 0.50)
        site_visits     = np.random.poisson(3) + 1
        product_pages   = np.random.randint(0, min(site_visits + 1, 10))
        days_to_visit   = np.random.randint(1, 15)   # days after sample to first site visit

        # Compute conversion probability (ground truth for label)
        p = (
            conv_rates[concern]
            + age_bonus[age]
            + email_ctr * 0.55
            + (site_visits / 15) * 0.12
            + (product_pages / 10) * 0.08
            + np.random.uniform(-0.06, 0.06)   # noise
        )
        p = float(np.clip(p, 0.02, 0.97))
        converted = int(np.random.random() < p)

        records.append({
            "customer_id":      f"RS-{10000 + i}",
            "skin_concern":     concern,
            "skin_type":        skin_type,
            "age_range":        age,
            "climate":          climate,
            "email_open_rate":  round(email_open_rate, 3),
            "email_ctr":        round(email_ctr, 3),
            "site_visits":      site_visits,
            "product_pages_viewed": product_pages,
            "days_to_first_visit":  days_to_visit,
            # Target variable: purchased full-size within 30-60 days
            "converted":        converted,
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals and engineer interaction features."""
    df = df.copy()

    # Engagement ratio: how many product pages per visit?
    df["pages_per_visit"] = (df["product_pages_viewed"] / df["site_visits"]).round(3)

    # Email quality score — opens AND clicks both matter
    df["email_engagement_score"] = (
        df["email_open_rate"] * 0.4 + df["email_ctr"] * 0.6
    ).round(3)

    # Encode categoricals
    le = LabelEncoder()
    for col in ["skin_concern", "skin_type", "age_range", "climate"]:
        df[col + "_enc"] = le.fit_transform(df[col])

    return df


FEATURE_COLS = [
    "skin_concern_enc",
    "skin_type_enc",
    "age_range_enc",
    "climate_enc",
    "email_open_rate",
    "email_ctr",
    "email_engagement_score",
    "site_visits",
    "product_pages_viewed",
    "pages_per_visit",
    "days_to_first_visit",
]

FEATURE_LABELS = [
    "Skin concern",
    "Skin type",
    "Age range",
    "Climate",
    "Email open rate",
    "Email CTR",
    "Email engagement score",
    "Site visits",
    "Product pages viewed",
    "Pages per visit",
    "Days to first visit",
]


# ──────────────────────────────────────────────
# 3. MODEL TRAINING  (XGBoost primary)
# ──────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    df = build_features(df)
    X = df[FEATURE_COLS]
    y = df["converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # handle class imbalance
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*50}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Did not convert", "Converted"]))

    return model, X_test, y_test, y_prob


# ──────────────────────────────────────────────
# 4. VISUALISATION — 4-panel summary figure
# ──────────────────────────────────────────────

def plot_results(model, X_test, y_test, y_prob):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Riversol — Propensity-to-Buy Model Results", fontsize=14, fontweight="bold")

    # Panel 1: Feature importance
    ax = axes[0, 0]
    imp = model.feature_importances_
    idx = np.argsort(imp)
    ax.barh([FEATURE_LABELS[i] for i in idx], imp[idx], color="#378ADD")
    ax.set_title("Feature importance (XGBoost gain)")
    ax.set_xlabel("Relative importance")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: ROC curve
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color="#378ADD", lw=2, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 3: Score distribution
    ax = axes[1, 0]
    ax.hist(y_prob[y_test == 0], bins=25, alpha=0.6, color="#E24B4A", label="Did not convert")
    ax.hist(y_prob[y_test == 1], bins=25, alpha=0.6, color="#1D9E75", label="Converted")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution by true label")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 4: Confusion matrix
    ax = axes[1, 1]
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        (y_prob >= 0.5).astype(int),
        display_labels=["No convert", "Convert"],
        colorbar=False,
        ax=ax,
        cmap="Blues",
    )
    ax.set_title("Confusion matrix (threshold = 0.50)")

    plt.tight_layout()
    plt.savefig("riversol_model_results.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved → riversol_model_results.png")
    plt.show()


# ──────────────────────────────────────────────
# 5. LEAD SCORING  — score any new batch of customers
# ──────────────────────────────────────────────

def score_leads(model, df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Add a conversion_score and priority column to a customer dataframe.
    In production: pass a fresh Shopify query result here.
    """
    df_new = build_features(df_new.copy())
    df_new["conversion_score"] = model.predict_proba(df_new[FEATURE_COLS])[:, 1].round(3)
    df_new["priority"] = pd.cut(
        df_new["conversion_score"],
        bins=[0, 0.35, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
    )
    return df_new.sort_values("conversion_score", ascending=False)


# ──────────────────────────────────────────────
# 6. SHAP EXPLAINABILITY  (why did model score this customer?)
# ──────────────────────────────────────────────

def explain_prediction(model, df_new: pd.DataFrame, customer_idx: int = 0):
    """
    Show which features drove the score for a specific customer.
    Useful for the marketing team to understand actionable signals.
    """
    df_new = build_features(df_new.copy())
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_new[FEATURE_COLS])

    print(f"\nExplaining prediction for customer: {df_new.iloc[customer_idx]['customer_id']}")
    print(f"  Conversion score: {df_new.iloc[customer_idx]['conversion_score']:.2%}")
    shap.initjs()
    shap.force_plot(
        explainer.expected_value,
        shap_values[customer_idx],
        df_new[FEATURE_COLS].iloc[customer_idx],
        feature_names=FEATURE_LABELS,
        matplotlib=True,
    )
    plt.title("SHAP force plot — why this customer scored high/low")
    plt.tight_layout()
    plt.savefig("riversol_shap_explanation.png", dpi=150, bbox_inches="tight")
    print("SHAP plot saved → riversol_shap_explanation.png")
    plt.show()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic Riversol customer data...")
    df = generate_synthetic_data(n=5000)
    print(f"  {len(df)} customers | conversion rate: {df['converted'].mean():.1%}\n")

    print("Training XGBoost propensity model...")
    model, X_test, y_test, y_prob = train_model(df)

    print("\nPlotting results...")
    plot_results(model, X_test, y_test, y_prob)

    print("\nScoring the top 10 leads...")
    new_batch = generate_synthetic_data(n=50)    # simulates a fresh weekly cohort
    scored    = score_leads(model, new_batch)
    print(scored[["customer_id", "skin_concern", "age_range",
                   "email_ctr", "site_visits", "conversion_score", "priority"]].head(10).to_string(index=False))

    print("\nExplaining top lead prediction (SHAP)...")
    explain_prediction(model, scored, customer_idx=0)
