import numpy as np


def apply_label_noise(y, noise_rate, random_state=42):
    """
    Apply symmetric label noise by flipping labels randomly.

    Parameters:
    - y: array-like of binary labels (0/1)
    - noise_rate: fraction of labels to flip (0.0 to 1.0)
    - random_state: seed for reproducibility

    Returns:
    - y_noisy: corrupted labels
    """
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()

    n_samples = len(y)
    n_noisy = int(noise_rate * n_samples)

    noisy_indices = rng.choice(n_samples, size=n_noisy, replace=False)
    y_noisy[noisy_indices] = 1 - y_noisy[noisy_indices]

    return y_noisy
