"""Загрузка sentiment-датасета из CSV."""

from pathlib import Path

import pandas as pd

DATASET_PATH = Path(__file__).resolve().parent / "sentiment_dataset.csv"
REQUIRED_COLUMNS = ("text", "label")


def load_sentiment_dataset(path=None):
    """Загружает CSV с колонками text и label."""
    csv_path = Path(path) if path is not None else DATASET_PATH
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV must contain columns {REQUIRED_COLUMNS}, missing: {missing}"
        )
    return df
