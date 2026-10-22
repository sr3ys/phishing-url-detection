"""
Phishing URL Detection - Baseline Model Script (Logistic Regression)

This script implements Step 5 of the project:
1. Loads the 18 lexical features from data/processed/features.csv
2. Splits the dataset into 80% training and 20% testing (stratified, random_state=42)
3. Standardizes features with StandardScaler
4. Trains a baseline Logistic Regression classifier
5. Evaluates and prints Accuracy, Precision, Recall, F1 Score, and Classification Report.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def run_baseline():
    dataset_path = os.path.join("data", "processed", "features.csv")

    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        print("Please run Step 4 (src/feature_extraction.py) first.")
        return

    print(f"Loading features dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Separate feature matrix X and target vector y
    feature_columns = [col for col in df.columns if col not in ["url", "label"]]
    X = df[feature_columns]
    y = df["label"]

    print(f"Total samples: {len(df):,}")
    print(f"Number of features: {len(feature_columns)}")

    # 80% Train, 20% Test Split with Stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Testing set:  {X_test.shape[0]:,} samples")

    # Feature Scaling (Standardization is essential for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Baseline Logistic Regression
    print("\nTraining Baseline Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Compute Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print(" BASELINE MODEL EVALUATION (Logistic Regression)")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Precision : {prec:.4f}  ({prec * 100:.2f}%)")
    print(f"  Recall    : {rec:.4f}  ({rec * 100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}  ({f1 * 100:.2f}%)")
    print("=" * 55)

    print("\nFull Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Legitimate (0)", "Phishing (1)"],
        digits=4
    ))


if __name__ == "__main__":
    run_baseline()
