import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_PATH = Path("results/label_noise_results.csv")
FIGURES_PATH = Path("figures")

FIGURES_PATH.mkdir(exist_ok=True)


def plot_label_noise_results():
    df = pd.read_csv(RESULTS_PATH)

    plt.figure()
    plt.plot(df["noise_rate"], df["accuracy"], marker="o", label="Accuracy")
    plt.plot(df["noise_rate"], df["roc_auc"], marker="s", label="ROC-AUC")

    plt.xlabel("Label Noise Rate")
    plt.ylabel("Performance")
    plt.title("Impact of Label Noise on Model Performance")
    plt.legend()
    plt.grid(True)

    output_path = FIGURES_PATH / "label_noise_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    plot_label_noise_results()
