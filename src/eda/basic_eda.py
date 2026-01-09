from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "adult_clean.csv"

df = pd.read_csv(DATA_PATH)

print("=== Dataset Shape ===")
print(df.shape)

print("\n=== Dataset Info ===")
print(df.info())

print("\n=== Income Distribution ===")
print(df["income"].value_counts(normalize=True))
