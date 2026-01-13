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
from src.data.corruptions.missingness import apply_missingness


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "missing_data_results.csv"


def run_missing_data_experiment(missing_rate, random_state):
    df = load_clean_dataset()

    X = df.drop(columns=[TARGET_COL])
    y = (df[TARGET_COL] == ">50K").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )

    X_train_missing = apply_missingness(
        X_train,
        missing_rate=missing_rate,
        random_state=random_state
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_missing)
    X_test_proc = preprocessor.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_proc, y_train)

    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1]

    return {
        "missing_rate": missing_rate,
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "random_state": random_state
    }


if __name__ == "__main__":
    results = []

    missing_rates = [0.0, 0.05, 0.10, 0.20]
    seeds = [0, 1, 2, 3, 4]

    for rate in missing_rates:
        for seed in seeds:
            res = run_missing_data_experiment(rate, seed)
            print(
                f"Missing: {rate:.2f} | "
                f"Seed: {seed} | "
                f"Acc: {res['accuracy']:.4f} | "
                f"AUC: {res['roc_auc']:.4f}"
            )
            results.append(res)

    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH.resolve()}")
