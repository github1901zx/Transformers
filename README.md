# Sentiment Analysis с BERT (7-дневный курс)

## Описание
Этот проект представляет собой завершенный цикл разработки системы анализа тональности текстов на основе архитектуры Transformers. В течение 7 дней мы прошли путь от основ токенизации до развертывания модели в виде API.

## Структура проекта
- `day_1/` — Основы токенизации (`AutoTokenizer`, BPE, WordPiece).
- `day_2/` — Работа с эмбеддингами и визуализация активаций.
- `day_3/` — Механизм Attention и визуализация карт внимания.
- `day_4/` — Baseline модель: Логистическая регрессия на CLS-эмбеддингах.
- `day_5/` — Fine-tuning: Дообучение DistilBERT на специфическом датасете.
- `day_6/` — Инференс и сравнение Baseline vs Fine-tuned моделей.
- `day_7/` — Анализ ошибок и создание API (FastAPI).
- `fine_tuned_model/` — Чекпоинт fine-tuned модели (config, tokenizer, `model.safetensors`).
- `baseline_model/` — Чекпоинт baseline (classifier, tokenizer, metadata с revision).
- `model_artifacts.py` — Проверка наличия сохранённых артефактов перед инференсом.
- `sentiment_dataset.csv` — CSV-датасет с колонками `text` и `label`.
- `api.py` — FastAPI приложение для предсказаний.
- `error_analysis.txt` — Результаты анализа сложных случаев для модели.

## Результаты

### Fine-tuned модель (DistilBERT):
- **F1 (macro):** см. `fine_tuned_results.txt` (оценка на уникальных текстах без утечки train/test)
- **Accuracy:** см. `fine_tuned_results.txt`

### Baseline модель (Logistic Regression + CLS):
- **F1 (macro):** см. `day_4/baseline_results.txt`
- **Accuracy:** см. `day_4/baseline_results.txt`

*Примечание: Метрики считаются на датасете из 60 уникальных отзывов (30 positive / 30 negative) с корректным разбиением train/test. Анализ ошибок в `error_analysis.txt` показывает, что модель может ошибаться на саркастичных или нейтрально-окрашенных фразах.*

## Артефакты моделей

Перед инференсом (день 6, API) должны быть сохранены оба чекпоинта:

```bash
python day_4/baseline_transformer.py   # baseline_model/
python day_5/fine_tuning.py            # fine_tuned_model/ + model.safetensors
python day_6/inference_comparison.py   # только загрузка, без обучения
```

`fine_tuned_model/model.safetensors` (~268 MB) хранится через Git LFS. После `git clone` выполните `git lfs pull`, иначе вместо весов останется LFS pointer.

Скрипт дня 6 **не переобучает** модели: при отсутствии артефактов выдаёт понятную ошибку с инструкцией.

## Запуск демо

### Вариант с FastAPI:
1. Установите зависимости:
   ```bash
   pip install fastapi uvicorn transformers torch
   ```
2. Запустите сервер:
   ```bash
   python api.py
   ```
   Или через uvicorn напрямую:
   ```bash
   uvicorn api:app --reload
   ```
3. API будет доступен на `http://127.0.0.1:8000`. Вы можете протестировать его через `/docs` (Swagger UI).

### Пример запроса к API:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "This project is amazing!"
}'
```

## Требования
- Python 3.8+
- PyTorch
- Transformers
- Scikit-learn
- FastAPI
- Uvicorn
- Matplotlib / Seaborn (для визуализаций)

---
*Поздравляем! Проект по изучению трансформеров завершен.*
