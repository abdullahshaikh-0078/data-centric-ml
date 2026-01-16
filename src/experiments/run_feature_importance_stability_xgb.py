import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.preprocess_adult import (
    load_clean_dataset,
    build_preprocessor,
    TARGET_COL
)
from src.data.corruptions.label_noise import apply_label_noise
from src.evaluation.feature_importance import compute_importance_stability


RESULTS_PATH = Path("results/feature_importance_stability_xgb.csv")


def run_experiment(noise_rate: float, seed: int) -> pd.Series:
    df = load_clean_dataset()

    X = df.drop(columns=[TARGET_COL])
    y = (df[TARGET_COL] == ">50K").astype(int)

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    # Apply label noise ONLY to training labels
    y_train_corrupt = apply_label_noise(
        y_train.values,
        noise_rate=noise_rate,
        random_state=seed
    )

    # Preprocessing (same pipeline)
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="logloss",
        use_label_encoder=False
    )

    model.fit(X_train_proc, y_train_corrupt)

    importance = model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    return pd.Series(importance, index=feature_names, name=f"seed_{seed}")


if __name__ == "__main__":
    noise_rate = 0.10
    seeds = [0, 1, 2, 3, 4]

    importance_runs = []

    for seed in seeds:
        imp = run_experiment(noise_rate, seed)
        importance_runs.append(imp)

    importance_df = pd.DataFrame(importance_runs)

    stability = compute_importance_stability(importance_df)

    print("XGBoost Feature Importance Stability")
    print(stability)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    importance_df.to_csv(RESULTS_PATH)

    print(f"\nSaved XGBoost importance vectors to {RESULTS_PATH.resolve()}")
