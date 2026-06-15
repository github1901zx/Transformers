#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Установка зависимостей"
python3 -m pip install -r requirements.txt

echo "==> Подтягивание Git LFS объектов"
if command -v git-lfs >/dev/null 2>&1; then
  git lfs install
  git lfs pull
else
  echo "WARNING: git-lfs не установлен. Установите его для загрузки model.safetensors."
fi

echo "==> Проверка чекпоинтов"
python3 - <<'PY'
from model_artifacts import require_fine_tuned_model, require_baseline_artifacts

require_fine_tuned_model()
require_baseline_artifacts()
print("Чекпоинты fine-tuned и baseline найдены.")
PY

echo "Готово."
