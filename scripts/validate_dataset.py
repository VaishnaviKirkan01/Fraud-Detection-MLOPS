
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place the PaySim CSV at data/raw/paysim.csv first."
    )

print("Checking dataset...")
df = pd.read_csv(DATA_PATH, nrows=100_000)

print("\nDataset sample shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows in inspected sample:", df.duplicated().sum())

if "isFraud" in df.columns:
    print("\nFraud class distribution in inspected sample:")
    print(df["isFraud"].value_counts(dropna=False))
    print("\nFraud class percentages:")
    print(
        (df["isFraud"].value_counts(normalize=True, dropna=False) * 100)
        .round(4)
    )
else:
    print("\nWARNING: Expected target column 'isFraud' is missing.")

print("\nInitial dataset inspection completed.")
