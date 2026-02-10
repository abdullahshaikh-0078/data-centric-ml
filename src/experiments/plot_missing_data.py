import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path("results/missing_data_results.csv")
FIGURES_PATH = Path("figures")
FIGURES_PATH.mkdir(exist_ok=True)

df = pd.read_csv(RESULTS_PATH)

summary = df.groupby("missing_rate").agg(
    acc_mean=("accuracy", "mean"),
    acc_std=("accuracy", "std"),
    auc_mean=("roc_auc", "mean"),
    auc_std=("roc_auc", "std"),
).reset_index()

plt.figure()
plt.errorbar(summary["missing_rate"], summary["acc_mean"], yerr=summary["acc_std"], marker="o", label="Accuracy")
plt.errorbar(summary["missing_rate"], summary["auc_mean"], yerr=summary["auc_std"], marker="s", label="ROC-AUC")

plt.xlabel("Missing Rate")
plt.ylabel("Performance")
plt.title("Impact of Missing Data on Model Performance")
plt.legend()
plt.grid(True)

plt.savefig(FIGURES_PATH / "missing_data_performance.png", dpi=300, bbox_inches="tight")
plt.close()
