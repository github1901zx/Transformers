"""Проверка и восстановление локальных артефактов моделей."""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FINE_TUNED_MODEL_DIR = PROJECT_ROOT / "fine_tuned_model"
WEIGHTS_FILE = FINE_TUNED_MODEL_DIR / "model.safetensors"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
MIN_WEIGHTS_BYTES = 1_000_000


def is_lfs_pointer(path):
    if not path.is_file():
        return False
    with path.open("rb") as file:
        header = file.read(len(LFS_POINTER_PREFIX))
    return header == LFS_POINTER_PREFIX


def has_valid_weights(path=WEIGHTS_FILE):
    if not path.is_file():
        return False
    if is_lfs_pointer(path):
        return False
    return path.stat().st_size >= MIN_WEIGHTS_BYTES


def _run_fine_tuning(save_path=FINE_TUNED_MODEL_DIR):
    script_path = PROJECT_ROOT / "day_5" / "fine_tuning.py"
    spec = importlib.util.spec_from_file_location("fine_tuning", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fine_tuning"] = module
    spec.loader.exec_module(module)
    return module.run_fine_tuning(save_path=save_path)


def ensure_fine_tuned_model(force_retrain=False):
    """Гарантирует наличие локальных весов fine-tuned модели."""
    if force_retrain or not has_valid_weights():
        if WEIGHTS_FILE.is_file() and is_lfs_pointer(WEIGHTS_FILE):
            print(
                "Обнаружен Git LFS pointer вместо весов модели. "
                "Запускаю детерминированный fine-tuning..."
            )
        elif not WEIGHTS_FILE.is_file():
            print("Веса fine-tuned модели не найдены. Запускаю fine-tuning...")
        else:
            print("Файл весов повреждён или неполный. Перезапускаю fine-tuning...")

        _run_fine_tuning(save_path=FINE_TUNED_MODEL_DIR)

    if not has_valid_weights():
        raise RuntimeError(
            f"Не удалось получить валидные веса модели в {WEIGHTS_FILE}. "
            "Запустите вручную: python day_5/fine_tuning.py"
        )

    return FINE_TUNED_MODEL_DIR
