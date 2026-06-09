import json
import sys
from pathlib import Path

import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentiment_data import load_sentiment_dataset
from model_config import MODEL_NAME, MODEL_REVISION, RANDOM_SEED

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = PROJECT_ROOT / "baseline_model"
RESULTS_PATH = PROJECT_ROOT / "day_4" / "baseline_results.txt"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)


def load_feature_extractor():
    model = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model.eval()
    return model


def tokenize_texts(texts, tokenizer, max_length=128):
    """Принимает список текстов и возвращает токенизированный батч (PyTorch tensors)."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def get_cls_embeddings(texts, model, tokenizer, batch_size=32):
    """Извлекает CLS-эмбеддинги для списка текстов."""
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenize_texts(batch_texts, tokenizer)

        with torch.no_grad():
            outputs = model(**inputs)

        cls_output = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_output)

    return np.vstack(embeddings)


def demo_tokenization(tokenizer):
    print("--- Тест ЗАДАЧИ 1: Токенизация ---")
    test_texts = ["Hello world!", "Transformers are awesome.", "Natural Language Processing."]
    tokenized_batch = tokenize_texts(test_texts, tokenizer)
    print(f"Keys in batch: {tokenized_batch.keys()}")
    print(f"Input IDs shape: {tokenized_batch['input_ids'].shape}")
    print("-" * 30 + "\n")
    return test_texts


def demo_embeddings(model, tokenizer, test_texts):
    print("--- Тест ЗАДАЧИ 2: Эмбеддинги ---")
    cls_embeddings = get_cls_embeddings(test_texts, model, tokenizer)
    print(f"Embeddings shape: {cls_embeddings.shape}")
    print("-" * 30 + "\n")


def save_baseline_artifacts(classifier, tokenizer):
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, BASELINE_DIR / "classifier.pkl")
    tokenizer.save_pretrained(BASELINE_DIR)

    metadata = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
    }
    with open(BASELINE_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Baseline артефакты сохранены в {BASELINE_DIR}")


def run_baseline(tokenizer=None, feature_extractor=None):
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if feature_extractor is None:
        feature_extractor = load_feature_extractor()

    df = load_sentiment_dataset()
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    print("Извлечение эмбеддингов для всего датасета...")
    X = get_cls_embeddings(texts, feature_extractor, tokenizer)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    lr_model = LogisticRegression(max_iter=1000, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print("--- Отчет о классификации ---")
    print(report)

    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1 Score: {f1:.4f}")

    save_baseline_artifacts(lr_model, tokenizer)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("Baseline Results (Logistic Regression on CLS embeddings)\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Revision: {MODEL_REVISION}\n")
        f.write(f"Macro F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"\nРезультаты сохранены в {RESULTS_PATH}")


def main():
    tokenizer = load_tokenizer()
    feature_extractor = load_feature_extractor()

    test_texts = demo_tokenization(tokenizer)
    demo_embeddings(feature_extractor, tokenizer, test_texts)
    run_baseline(tokenizer=tokenizer, feature_extractor=feature_extractor)


if __name__ == "__main__":
    main()
