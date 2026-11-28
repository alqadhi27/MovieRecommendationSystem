from pathlib import Path
import pandas as pd

class DataLoader:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)

    def prepare(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"File not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        df = self._clean(df)
        df["genres_list"] = df["genres"].apply(self._split_list)
        df["keywords_list"] = df["keywords"].apply(self._split_list)
        df["directors_list"] = df["directors"].apply(self._split_list)
        df = self._build_text_features(df)
        return df

    def _clean(self, df):
        df = df.copy()
        df["runtime"] = df["runtime"].replace(0, pd.NA).astype("Int64")
        dates = pd.to_datetime(df["release_date"], errors="coerce")
        df["year"] = dates.dt.year.astype("Int64")
        return df

    def _split_list(self, value):
        if pd.isna(value):
            return []
        parts = value.split(",")
        return [part.strip() for part in parts if part.strip()]

    def _build_text_features(self, df):
        df = df.copy()
        def combine_row(row):
            overview = "" if pd.isna(row["overview"]) else str(row["overview"]).strip()
            genres = " ".join(row["genres_list"])
            keywords = " ".join(row["keywords_list"])
            directors = " ".join(name.replace(" ", "") for name in row["directors_list"])
            parts = [overview, genres, keywords, directors]
            return " ".join(part for part in parts if part)
        df["text"] = df.apply(combine_row, axis=1)
        return df
