import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_PATH = Path("results/label_noise_results.csv")
FIGURES_PATH = Path("figures")
FIGURES_PATH.mkdir(exist_ok=True)


def plot_label_noise_results():
    df = pd.read_csv(RESULTS_PATH)

    # Aggregate across random seeds
    summary = df.groupby("noise_rate").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        auc_mean=("roc_auc", "mean"),
        auc_std=("roc_auc", "std"),
    ).reset_index()

    plt.figure()
    plt.errorbar(
        summary["noise_rate"],
        summary["accuracy_mean"],
        yerr=summary["accuracy_std"],
        marker="o",
        label="Accuracy"
    )
    plt.errorbar(
        summary["noise_rate"],
        summary["auc_mean"],
        yerr=summary["auc_std"],
        marker="s",
        label="ROC-AUC"
    )

    plt.xlabel("Label Noise Rate")
    plt.ylabel("Performance")
    plt.title("Impact of Label Noise on Model Performance")
    plt.legend()
    plt.grid(True)

    output_path = FIGURES_PATH / "label_noise_performance_mean_std.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_path.resolve()}")


if __name__ == "__main__":
    plot_label_noise_results()
