import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.evaluation.feature_importance import compute_importance_stability


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def load_and_compute(path: Path):
    df = pd.read_csv(path, index_col=0)
    return compute_importance_stability(df)


if __name__ == "__main__":
    lr_path = RESULTS_DIR / "feature_importance_stability.csv"
    rf_path = RESULTS_DIR / "feature_importance_stability_rf.csv"
    xgb_path = RESULTS_DIR / "feature_importance_stability_xgb.csv"

    lr_stats = load_and_compute(lr_path)
    rf_stats = load_and_compute(rf_path)
    xgb_stats = load_and_compute(xgb_path)

    models = ["Logistic Regression", "Random Forest", "XGBoost"]
    means = [
        lr_stats["stability_mean"],
        rf_stats["stability_mean"],
        xgb_stats["stability_mean"],
    ]
    stds = [
        lr_stats["stability_std"],
        rf_stats["stability_std"],
        xgb_stats["stability_std"],
    ]

    plt.figure(figsize=(7, 4))
    plt.bar(models, means, yerr=stds, capsize=6)
    plt.ylabel("Spearman Rank Correlation")
    plt.title("Feature Importance Stability Across Models")
    plt.ylim(0, 1)

    output_path = FIGURES_DIR / "feature_importance_stability_all_models.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print("LR stability:", lr_stats)
    print("RF stability:", rf_stats)
    print("XGBoost stability:", xgb_stats)
    print(f"\nSaved plot to {output_path.resolve()}")
