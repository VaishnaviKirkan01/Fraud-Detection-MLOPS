from __future__ import annotations

from pathlib import Path
import pandas as pd

try:
    # Works when run from the project root as a module/script.
    from ml.data.validation import validate_raw_dataframe
except ModuleNotFoundError:
    # Fallback for direct execution from within the ml/data folder.
    from validation import validate_raw_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"


def load_dataset(path: str | Path = DATA_PATH, validate: bool = True) -> pd.DataFrame:
    """Load the raw PaySim CSV and optionally validate it.

    This loads the complete dataset into memory. For quick checks on very large
    CSV files, use pd.read_csv(..., chunksize=...) instead.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the PaySim CSV at <project-root>/data/raw/paysim.csv."
        )

    df = pd.read_csv(path, low_memory=False)

    if validate:
        validate_raw_dataframe(df)

    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print useful dataset summary information."""
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\nColumn data types:")
    print(df.dtypes.to_string())
    print("\nMissing values by column:")
    print(df.isna().sum().to_string())
    print("\nFraud label counts:")
    print(df["isFraud"].value_counts(dropna=False).sort_index().to_string())
    print("\nFraud label percentages:")
    print(
        (df["isFraud"].value_counts(normalize=True, dropna=False) * 100)
        .sort_index()
        .round(4)
        .to_string()
    )
    print("\nTransaction types:")
    print(df["type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    dataset = load_dataset()
    print_dataset_summary(dataset)
