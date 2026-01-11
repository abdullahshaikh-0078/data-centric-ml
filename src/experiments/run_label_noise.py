from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data.preprocess_adult import load_clean_dataset, build_preprocessor, TARGET_COL
from src.data.corruptions.label_noise import apply_label_noise


def run_label_noise_experiment(noise_rate):
    df = load_clean_dataset()

    X = df.drop(columns=[TARGET_COL])
    y = (df[TARGET_COL] == ">50K").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply label noise ONLY to training labels
    y_train_noisy = apply_label_noise(
        y_train.values, noise_rate=noise_rate, random_state=42
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_processed, y_train_noisy)

    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Noise rate: {noise_rate:.2f} | Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    for noise in [0.0, 0.05, 0.10, 0.20]:
        run_label_noise_experiment(noise)
