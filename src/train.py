"""
Phishing URL Detection - Multi-Model Training and Comparison Module

This script implements Step 6:
1. Loads features from data/processed/features.csv
2. Splits dataset (80% train, 20% test, stratified, random_state=42)
3. Trains and benchmarks 4 ML algorithms:
   - Logistic Regression
   - Decision Tree
   - Random Forest (Multi-core CPU)
   - XGBoost / Gradient Boosting (GPU-accelerated with CUDA)
4. Records Accuracy, Precision, Recall, F1 Score, and Training Time
5. Displays a comparison table and saves a comparison bar chart to reports/figures/models_comparison.png
6. Performs light hyperparameter tuning on the best model via GridSearchCV.
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Try importing XGBoost for GPU-accelerated gradient boosting
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import GradientBoostingClassifier


def train_and_compare_models():
    dataset_path = os.path.join("data", "processed", "features.csv")
    figures_dir = os.path.join("reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)

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

    print(f"Features: {len(feature_cols)} lexical features")
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Testing samples:  {X_test.shape[0]:,}\n")

    # Scale data for algorithms that require normalization (e.g., Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the 4 models
    # Check if GPU is available for XGBoost
    xgb_device = "cuda" if HAS_XGBOOST else "cpu"

    models_dict = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "use_scaled": True
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=15, random_state=42),
            "use_scaled": False
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "use_scaled": False
        }
    }

    if HAS_XGBOOST:
        models_dict["XGBoost (GPU)"] = {
            "model": xgb.XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                device=xgb_device,
                random_state=42,
                eval_metric="logloss"
            ),
            "use_scaled": False
        }
    else:
        models_dict["Gradient Boosting"] = {
            "model": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "use_scaled": False
        }

    results = []
    fitted_models = {}

    print("=" * 70)
    print(" TRAINING AND BENCHMARKING MODELS")
    print("=" * 70)

    for name, config in models_dict.items():
        clf = config["model"]
        use_scaled = config["use_scaled"]

        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        print(f"Training {name}...")
        start_time = time.time()
        clf.fit(X_tr, y_train)
        train_time = time.time() - start_time

        # Predict on the unseen 20% test set
        y_pred = clf.predict(X_te)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "Train Time (s)": round(train_time, 2)
        })
        fitted_models[name] = clf

    # Convert results into a neat DataFrame
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print(" MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 70)
    display_df = results_df.copy()
    display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["Precision"] = display_df["Precision"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["Recall"] = display_df["Recall"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["F1 Score"] = display_df["F1 Score"].apply(lambda x: f"{x * 100:.2f}%")
    display_df["Train Time (s)"] = display_df["Train Time (s)"].apply(lambda x: f"{x:.2f}s")
    print(display_df.to_string(index=False))
    print("=" * 70)

    # -------------------------------------------------------------
    # Generate and Save Comparison Bar Chart
    # -------------------------------------------------------------
    print("\nGenerating model comparison chart...")
    metrics_melted = pd.melt(
        results_df,
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    palette = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    ax = sns.barplot(
        data=metrics_melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette=palette
    )

    plt.title(f"Model Performance Comparison (Test Set: {len(X_test):,} URLs)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Evaluation Metric", fontsize=12, labelpad=10)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12, labelpad=10)
    plt.ylim(0.65, 1.00)
    plt.legend(title="Model", loc="lower right", frameon=True)

    # Add numeric percentage labels above each bar
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height*100:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points"
            )

    plt.tight_layout()
    chart_path = os.path.join(figures_dir, "models_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Comparison chart saved to: {chart_path}")

    # -------------------------------------------------------------
    # Hyperparameter Tuning on Best Model (XGBoost / Random Forest)
    # -------------------------------------------------------------
    best_candidate_name = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]["Model"]
    print(f"\nTop performing model identified: {best_candidate_name}")
    print("Starting light hyperparameter tuning via GridSearchCV...")

    if HAS_XGBOOST and "XGBoost" in best_candidate_name:
        param_grid = {
            "n_estimators": [100, 150],
            "max_depth": [6, 8],
            "learning_rate": [0.1, 0.2]
        }
        base_xgb = xgb.XGBClassifier(
            device=xgb_device,
            random_state=42,
            eval_metric="logloss"
        )
        grid_search = GridSearchCV(
            estimator=base_xgb,
            param_grid=param_grid,
            cv=3,
            scoring="f1",
            verbose=1,
            n_jobs=1  # XGBoost GPU manages its own internal parallelism
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        param_grid = {
            "n_estimators": [100, 150],
            "max_depth": [12, 16],
            "min_samples_split": [2, 5]
        }
        base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=param_grid,
            cv=3,
            scoring="f1",
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

    print(f"\nTuning Complete! Best parameters: {best_params}")

    # Final evaluation of tuned best model
    y_pred_best = best_model.predict(X_test)
    tuned_acc = accuracy_score(y_test, y_pred_best)
    tuned_prec = precision_score(y_test, y_pred_best)
    tuned_rec = recall_score(y_test, y_pred_best)
    tuned_f1 = f1_score(y_test, y_pred_best)

    print("\n" + "=" * 70)
    print(f" FINAL TUNED BEST MODEL ({best_candidate_name}) PERFORMANCE")
    print("=" * 70)
    print(f"  Accuracy  : {tuned_acc:.4f} ({tuned_acc*100:.2f}%)")
    print(f"  Precision : {tuned_prec:.4f} ({tuned_prec*100:.2f}%)")
    print(f"  Recall    : {tuned_rec:.4f} ({tuned_rec*100:.2f}%)")
    print(f"  F1 Score  : {tuned_f1:.4f} ({tuned_f1*100:.2f}%)")
    print("=" * 70)

    return results_df, best_model, best_params


if __name__ == "__main__":
    train_and_compare_models()
