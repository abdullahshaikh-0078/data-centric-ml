from pathlib import Path
import pandas as pd

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "adult_clean.csv"

df = pd.read_csv(DATA_PATH)

print("\n=== Numerical Feature Summary ===")
num_cols = df.select_dtypes(include=["int64"]).columns
print(df[num_cols].describe())

print("\n=== Capital Gain / Loss Sparsity ===")
print("capital_gain > 0 proportion:", (df["capital_gain"] > 0).mean())
print("capital_loss > 0 proportion:", (df["capital_loss"] > 0).mean())

print("\n=== Categorical Cardinality ===")
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} unique values")

print("\n=== Top Categories (native_country) ===")
print(df["native_country"].value_counts(normalize=True).head(10))

print("\n=== Income by Sex ===")
print(pd.crosstab(df["sex"], df["income"], normalize="index"))

print("\n=== Income by Race ===")
print(pd.crosstab(df["race"], df["income"], normalize="index"))
