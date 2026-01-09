import pandas as pd


def load_adult_dataset(path: str) -> pd.DataFrame:
    """
    Load the Adult Income dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the dataset CSV file

    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    data_path = "data/raw/adult_clean.csv"
    df = load_adult_dataset(data_path)

    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    