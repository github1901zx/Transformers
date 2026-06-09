"""Проверка наличия сохранённых артефактов моделей."""

import json
from pathlib import Path

import joblib
from transformers import AutoModel, AutoTokenizer

from model_config import MODEL_NAME, MODEL_REVISION

PROJECT_ROOT = Path(__file__).resolve().parent
FINE_TUNED_MODEL_DIR = PROJECT_ROOT / "fine_tuned_model"
BASELINE_MODEL_DIR = PROJECT_ROOT / "baseline_model"
WEIGHTS_FILE = FINE_TUNED_MODEL_DIR / "model.safetensors"
PYTORCH_WEIGHTS_FILE = FINE_TUNED_MODEL_DIR / "pytorch_model.bin"
CLASSIFIER_FILE = BASELINE_MODEL_DIR / "classifier.pkl"
METADATA_FILE = BASELINE_MODEL_DIR / "metadata.json"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
MIN_WEIGHTS_BYTES = 1_000_000


def is_lfs_pointer(path):
    if not path.is_file():
        return False
    with path.open("rb") as file:
        header = file.read(len(LFS_POINTER_PREFIX))
    return header == LFS_POINTER_PREFIX


def _weights_path():
    if WEIGHTS_FILE.is_file():
        return WEIGHTS_FILE
    if PYTORCH_WEIGHTS_FILE.is_file():
        return PYTORCH_WEIGHTS_FILE
    return None


def has_valid_weights():
    path = _weights_path()
    if path is None:
        return False
    if is_lfs_pointer(path):
        return False
    return path.stat().st_size >= MIN_WEIGHTS_BYTES


def require_fine_tuned_model():
    """Проверяет, что fine-tuned чекпоинт содержит веса модели."""
    if not has_valid_weights():
        path = _weights_path()
        if path is not None and is_lfs_pointer(path):
            reason = (
                "обнаружен Git LFS pointer вместо весов. "
                "Выполните: git lfs pull"
            )
        else:
            reason = "файл весов отсутствует"

        raise FileNotFoundError(
            f"Fine-tuned модель не готова к загрузке ({reason}).\n"
            f"Ожидается model.safetensors или pytorch_model.bin в {FINE_TUNED_MODEL_DIR}.\n"
            "Сначала обучите и сохраните модель: python day_5/fine_tuning.py"
        )
    return FINE_TUNED_MODEL_DIR


def has_baseline_artifacts():
    if not CLASSIFIER_FILE.is_file() or not METADATA_FILE.is_file():
        return False
    try:
        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    required = {"model_name", "model_revision"}
    if not required.issubset(metadata):
        return False
    tokenizer_config = BASELINE_MODEL_DIR / "tokenizer_config.json"
    return tokenizer_config.is_file()


def require_baseline_artifacts():
    """Проверяет, что baseline сохранён как единый набор артефактов."""
    if not has_baseline_artifacts():
        raise FileNotFoundError(
            f"Baseline артефакты не найдены в {BASELINE_MODEL_DIR}.\n"
            "Ожидаются classifier.pkl, metadata.json и файлы токенизатора.\n"
            "Сначала обучите и сохраните baseline: python day_4/baseline_transformer.py"
        )
    return BASELINE_MODEL_DIR


def load_baseline_artifacts(device):
    """Загружает classifier, tokenizer и feature extractor из сохранённого baseline."""
    baseline_dir = require_baseline_artifacts()
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

    classifier = joblib.load(CLASSIFIER_FILE)
    tokenizer = AutoTokenizer.from_pretrained(baseline_dir)
    feature_extractor = AutoModel.from_pretrained(
        metadata["model_name"],
        revision=metadata["model_revision"],
    )
    feature_extractor.eval()
    feature_extractor.to(device)

    return classifier, feature_extractor, tokenizer, metadata
