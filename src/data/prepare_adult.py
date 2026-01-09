import pandas as pd
from pathlib import Path

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]


def prepare_adult_dataset(raw_csv_path: Path, output_csv_path: Path):
    df = pd.read_csv(
        raw_csv_path,
        header=None,
        names=COLUMN_NAMES,
        skipinitialspace=True
    )

    df.to_csv(output_csv_path, index=False)
    print("Clean dataset saved to:", output_csv_path)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    RAW_PATH = PROJECT_ROOT / "data" / "raw" / "adult.csv"
    OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "adult_clean.csv"

    prepare_adult_dataset(RAW_PATH, OUTPUT_PATH)
