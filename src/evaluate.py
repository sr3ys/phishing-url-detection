"""
Phishing URL Detection - Detailed Model Evaluation Module

This script implements Step 7:
1. Evaluates the best tuned model on the 20% unseen test set (10,000 URLs)
2. Computes and plots the Confusion Matrix heatmap -> reports/figures/confusion_matrix.png
3. Computes and plots the ROC Curve with AUC score -> reports/figures/roc_curve.png
4. Computes and plots Feature Importance -> reports/figures/feature_importance.png
5. Prints the complete classification report and plain-language security explanation
6. Saves the best model bundle to models/best_model.joblib
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestClassifier


def evaluate_best_model():
    dataset_path = os.path.join("data", "processed", "features.csv")
    figures_dir = os.path.join("reports", "figures")
    models_dir = os.path.join("models")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing {dataset_path}. Please complete Step 4 first.")

    print(f"Loading feature dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    feature_cols = [c for c in df.columns if c not in ["url", "label"]]
    X = df[feature_cols]
    y = df["label"]

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training tuned best model (XGBoost on GPU)...")
    if HAS_XGBOOST:
        best_model = xgb.XGBClassifier(
            device="cuda",
            n_estimators=100,
            max_depth=8,
            learning_rate=0.2,
            random_state=42,
            eval_metric="logloss"
        )
        model_name = "Tuned XGBoost (GPU Accelerated)"
    else:
        best_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            random_state=42,
            n_jobs=-1
        )
        model_name = "Tuned Random Forest"

    best_model.fit(X_train, y_train)

    # Predictions & Probabilities
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 65)
    print(f" {model_name.upper()} - EVALUATION REPORT")
    print("=" * 65)
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall    : {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f} ({f1*100:.2f}%)")
    print(f"  ROC-AUC   : {auc:.4f} ({auc*100:.2f}%)")
    print("=" * 65)

    print("\nDetailed Classification Report:")
    report_str = classification_report(
        y_test, y_pred,
        target_names=["Legitimate (0)", "Phishing (1)"],
        digits=4
    )
    print(report_str)

    # -------------------------------------------------------------
    # 1. Confusion Matrix Heatmap
    # -------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("Confusion Matrix Raw Counts:")
    print(f"  True Negatives  (Legitimate correctly classified): {tn:,}")
    print(f"  False Positives (Legitimate flagged as Phishing) : {fp:,}")
    print(f"  False Negatives (Phishing missed by model)       : {fn:,}")
    print(f"  True Positives  (Phishing correctly caught)      : {tp:,}")

    plt.figure(figsize=(7, 6))
    sns.set_theme(style="white")
    cm_labels = np.array([
        [f"True Negative\n{tn:,}", f"False Positive\n{fp:,}"],
        [f"False Negative\n{fn:,}", f"True Positive\n{tp:,}"]
    ])
    sns.heatmap(
        cm,
        annot=cm_labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted Legit (0)", "Predicted Phish (1)"],
        yticklabels=["Actual Legit (0)", "Actual Phish (1)"],
        annot_kws={"size": 11, "fontweight": "bold"}
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Actual True Category", fontsize=11)
    plt.xlabel("Model Prediction", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\n[Saved] Confusion matrix chart -> {cm_path}")

    # -------------------------------------------------------------
    # 2. ROC Curve and AUC Score
    # -------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    plt.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"{model_name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title(f"Receiver Operating Characteristic (ROC) Curve\nArea Under Curve (AUC) = {auc:.4f}", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()
    roc_path = os.path.join(figures_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Saved] ROC curve chart -> {roc_path}")

    # -------------------------------------------------------------
    # 3. Feature Importance Plot
    # -------------------------------------------------------------
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values(by="Importance", ascending=True)

        plt.figure(figsize=(10, 7))
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(feat_df)))
        plt.barh(feat_df["Feature"], feat_df["Importance"], color=colors)
        plt.title(f"URL Lexical Feature Importances ({model_name})", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Relative Importance Score", fontsize=11)
        plt.ylabel("Lexical Feature", fontsize=11)

        for i, val in enumerate(feat_df["Importance"]):
            plt.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=9)

        plt.xlim(0, max(feat_df["Importance"]) * 1.15)
        plt.tight_layout()
        feat_path = os.path.join(figures_dir, "feature_importance.png")
        plt.savefig(feat_path, dpi=300)
        plt.close()
        print(f"[Saved] Feature importance chart -> {feat_path}")

    # -------------------------------------------------------------
    # 4. Save Model Bundle to models/best_model.joblib
    # -------------------------------------------------------------
    model_bundle = {
        "model": best_model,
        "feature_names": feature_cols,
        "model_name": model_name,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc
        }
    }
    model_save_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(model_bundle, model_save_path)
    print(f"\n[Saved] Best model bundle saved to: {model_save_path}")

    # Plain Language Summary Explanation
    total_legit = tn + fp
    total_phish = fn + tp
    fp_pct = (fp / total_legit * 100) if total_legit > 0 else 0
    fn_pct = (fn / total_phish * 100) if total_phish > 0 else 0

    print("\n" + "=" * 65)
    print(" PLAIN LANGUAGE SECURITY EXPLANATION (For Presentation/Viva)")
    print("=" * 65)
    print(f"1. FALSE POSITIVES (FP = {fp:,} out of {total_legit:,} legitimate URLs, {fp_pct:.2f}%):")
    print("   A legitimate URL is incorrectly flagged as phishing.")
    print("   In the real world, this causes a minor user annoyance (warning prompt),")
    print("   which the user can bypass after checking.")
    print(f"\n2. FALSE NEGATIVES (FN = {fn:,} out of {total_phish:,} phishing URLs, {fn_pct:.2f}%):")
    print("   A phishing URL slips past the system as legitimate.")
    print("   In cyber security, this is HIGH RISK because the user trusts the link,")
    print("   potentially entering passwords or financial data into a fake site.")
    print(f"\n3. HIGH RECALL ({rec*100:.2f}%) & AUC ({auc*100:.2f}%):")
    print(f"   Our model catches {rec*100:.2f}% of phishing attacks while maintaining {prec*100:.2f}% precision,")
    print("   providing a reliable defense using fast offline lexical analysis.")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_best_model()
