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
from src.data.corruptions.label_noise import apply_label_noise


RESULTS_PATH = Path("results/label_noise_results.csv")


def run_label_noise_experiment(noise_rate, random_state=42):
    df = load_clean_dataset()

    X = df.drop(columns=[TARGET_COL])
    y = (df[TARGET_COL] == ">50K").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    # Apply label noise ONLY to training labels
    y_train_noisy = apply_label_noise(
        y_train.values,
        noise_rate=noise_rate,
        random_state=random_state
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

    return {
        "noise_rate": noise_rate,
        "accuracy": acc,
        "roc_auc": auc,
        "random_state": random_state
    }


if __name__ == "__main__":
    results = []

    for noise in [0.0, 0.05, 0.10, 0.20]:
        res = run_label_noise_experiment(noise)
        print(
            f"Noise rate: {noise:.2f} | "
            f"Accuracy: {res['accuracy']:.4f} | "
            f"ROC-AUC: {res['roc_auc']:.4f}"
        )
        results.append(res)

    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}")
