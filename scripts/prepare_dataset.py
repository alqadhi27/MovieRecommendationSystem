
from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "movies_cleaned.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "movies_filtered.csv"


# Required columns
REQUIRED_COLUMNS = [
    "id",
    "title",
    "poster_path",
    "release_date",
    "overview",
    "genres",
    "keywords",
    "directors",
    "original_language",
    "runtime",
    "vote_average",
    "vote_count",
]


def prepare_dataset():
    # Stop if the output file already exists
    if OUTPUT_FILE.exists():
        print("The filtered dataset already exists.")
        print("Delete it manually if you want to generate it again.")
        return

    # Check whether the input file exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    # Check required columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    initial_count = len(df)

    # Filter movies with at least 100 votes
    filtered_df = df.loc[
        df["vote_count"] >= 100,
        REQUIRED_COLUMNS
    ].copy()

    # Save the filtered dataset
    filtered_df.to_csv(OUTPUT_FILE, index=False)

    # Display results
    print("\nDataset preparation completed.")
    print(f"Original movies: {initial_count:,}")
    print(f"Filtered movies: {len(filtered_df):,}")
    print(f"Number of columns: {len(filtered_df.columns)}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    prepare_dataset()
