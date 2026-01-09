import pandas as pd
from pathlib import Path


def inspect_dataset(df: pd.DataFrame):
    print("=== Dataset Info ===")
    print(df.info())

    print("\n=== Missing Values (per column) ===")
    print(df.isnull().sum())

    print("\n=== Categorical Columns ===")
    categorical_cols = df.select_dtypes(include=["object"]).columns
    print(list(categorical_cols))

    print("\n=== Numerical Columns ===")
    numerical_cols = df.select_dtypes(exclude=["object"]).columns
    print(list(numerical_cols))


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = PROJECT_ROOT / "data" / "raw" / "adult_clean.csv"

    df = pd.read_csv(DATA_PATH)
    inspect_dataset(df)
