import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


NUMERICAL_COLS = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week"
]

CATEGORICAL_COLS = [
    "workclass", "education", "marital_status",
    "occupation", "relationship", "race",
    "sex", "native_country"
]

TARGET_COL = "income"


def load_clean_dataset():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "adult_clean.csv"
    return pd.read_csv(data_path)


def build_preprocessor():
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    return preprocessor


if __name__ == "__main__":
    df = load_clean_dataset()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)

    print("Processed training shape:", X_train_processed.shape)
