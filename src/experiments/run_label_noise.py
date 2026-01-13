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


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_PATH = RESULTS_DIR / "label_noise_results.csv"


# ------------------------------------------------------------------
# Core experiment function
# ------------------------------------------------------------------
def run_label_noise_experiment(noise_rate, random_state=42):
    """
    Train and evaluate logistic regression under label noise.

    Parameters
    ----------
    noise_rate : float
        Fraction of training labels to flip.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Experiment results.
    """
    # Load clean dataset
    df = load_clean_dataset()

    X = df.drop(columns=[TARGET_COL])
    y = (df[TARGET_COL] == ">50K").astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )

    # Apply label noise ONLY to training labels
    y_train_noisy = apply_label_noise(
        y=y_train.values,
        noise_rate=noise_rate,
        random_state=random_state
    )

    # Preprocessing
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_processed, y_train_noisy)

    # Evaluation
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


# ------------------------------------------------------------------
# Run experiment across noise levels and seeds
# ------------------------------------------------------------------
if __name__ == "__main__":
    results = []

    noise_levels = [0.0, 0.05, 0.10, 0.20]
    seeds = [0, 1, 2, 3, 4]

    print("=== Running Label Noise Robustness Experiments ===\n")

    for noise in noise_levels:
        for seed in seeds:
            res = run_label_noise_experiment(
                noise_rate=noise,
                random_state=seed
            )

            print(
                f"Noise: {noise:.2f} | "
                f"Seed: {seed} | "
                f"Accuracy: {res['accuracy']:.4f} | "
                f"ROC-AUC: {res['roc_auc']:.4f}"
            )

            results.append(res)

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_PATH, index=False)

    print(f"\nSaved results to: {RESULTS_PATH.resolve()}")
