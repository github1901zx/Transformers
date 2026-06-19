"""Загрузка sentiment-датасета из CSV."""

from pathlib import Path

import pandas as pd

DATASET_PATH = Path(__file__).resolve().parent / "sentiment_dataset.csv"
REQUIRED_COLUMNS = ("text", "label")
DEFAULT_LABEL_NAMES = {0: "Negative", 1: "Positive", 2: "Neutral"}


def get_num_labels(labels):
    return len(set(labels))


def get_target_names(labels):
    classes = sorted(set(labels))
    if classes != list(range(len(classes))):
        raise ValueError(
            f"Labels must be contiguous integers 0..{len(classes) - 1}, got {classes}"
        )
    return [DEFAULT_LABEL_NAMES.get(i, f"Class_{i}") for i in classes]


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
