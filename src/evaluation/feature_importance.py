import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_importance_stability(importance_df: pd.DataFrame):
    """
    Computes pairwise Spearman correlation between importance vectors
    across different runs (seeds).

    Rows = runs
    Columns = features
    """

    runs = importance_df.index.tolist()
    correlations = []

    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            corr, _ = spearmanr(
                importance_df.loc[runs[i]],
                importance_df.loc[runs[j]]
            )
            correlations.append(corr)

    correlations = np.array(correlations)
    correlations = correlations[~np.isnan(correlations)]

    return {
        "stability_mean": float(np.mean(correlations)),
        "stability_std": float(np.std(correlations))
    }
