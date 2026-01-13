import numpy as np
import pandas as pd


def apply_missingness(X, missing_rate, random_state=42):
    """
    Introduce Missing Completely At Random (MCAR) missingness
    using np.nan (compatible with scikit-learn).
    """
    rng = np.random.default_rng(random_state)
    X_missing = X.copy()

    n_rows, n_cols = X_missing.shape
    n_missing = int(missing_rate * n_rows * n_cols)

    rows = rng.integers(0, n_rows, size=n_missing)
    cols = rng.integers(0, n_cols, size=n_missing)

    for r, c in zip(rows, cols):
        X_missing.iat[r, c] = np.nan

    return X_missing
