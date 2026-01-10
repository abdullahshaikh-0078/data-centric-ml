import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data.preprocess_adult import (
    load_clean_dataset,
    build_preprocessor,
    TARGET_COL
)


def run_baseline():
    # Load clean dataset
    df = load_clean_dataset()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Convert labels to binary
    y = (y == ">50K").astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Preprocessing
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Logistic Regression baseline
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X_train_processed, y_train)

    # Predictions
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("=== Baseline Logistic Regression ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")


if __name__ == "__main__":
    run_baseline()
