import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentiment_data import get_labeled_data

# ==========================================
# ЗАДАЧА 1: Токенизация текстов
# ==========================================

# 1. & 2. Загрузка токенизатора
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Функция токенизации
def tokenize_texts(texts, max_length=128):
    """
    Принимает список текстов и возвращает токенизированный батч (PyTorch tensors).
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# 4. Тестирование ЗАДАЧИ 1
print("--- Тест ЗАДАЧИ 1: Токенизация ---")
test_texts = ["Hello world!", "Transformers are awesome.", "Natural Language Processing."]
tokenized_batch = tokenize_texts(test_texts)
print(f"Keys in batch: {tokenized_batch.keys()}")
print(f"Input IDs shape: {tokenized_batch['input_ids'].shape}")
print("-" * 30 + "\n")


# ==========================================
# ЗАДАЧА 2: Извлечение CLS-эмбеддингов
# ==========================================

# 2. & 3. Загрузка модели и перевод в eval
model = AutoModel.from_pretrained(model_name)
model.eval()

# 4. Функция извлечения CLS-эмбеддингов
def get_cls_embeddings(texts, batch_size=32):
    """
    Извлекает CLS-эмбеддинги для списка текстов.
    """
    embeddings = []
    
    # Разбиваем на батчи
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Токенизация
        inputs = tokenize_texts(batch_texts)
        
        # Прогон через модель без градиентов
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Извлечение CLS-токена (индекс 0)
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        cls_output = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_output)
        
    return np.vstack(embeddings)

# 5. Тестирование ЗАДАЧИ 2
print("--- Тест ЗАДАЧИ 2: Эмбеддинги ---")
cls_embeddings = get_cls_embeddings(test_texts)
print(f"Embeddings shape: {cls_embeddings.shape}")
print("-" * 30 + "\n")


# ==========================================
# ЗАДАЧА 3: Logistic Regression на эмбеддингах
# ==========================================

def run_baseline():
    # 1. Подготовка датасета (60 уникальных текстов, без дубликатов)
    df = pd.DataFrame(get_labeled_data())

    # 2. Извлечение текстов и меток
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # 3. Получение эмбеддингов
    print("Извлечение эмбеддингов для всего датасета...")
    X = get_cls_embeddings(texts)
    y = np.array(labels)

    # 4. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 6. Создание модели
    lr_model = LogisticRegression(max_iter=1000, n_jobs=-1)

    # 7. Обучение
    lr_model.fit(X_train, y_train)

    # 8. Предсказание
    y_pred = lr_model.predict(X_test)

    # 10. Отчет
    report = classification_report(y_test, y_pred)
    print("--- Отчет о классификации ---")
    print(report)

    # 11. Macro F1
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 Score: {f1:.4f}")

    # 11.5 Сохранение модели для Дня 6
    import joblib
    joblib.dump(lr_model, 'baseline_model.pkl')
    # В этой реализации vectorizer нет, так как мы используем эмбеддинги трансформера.
    # Но в Задачах Дня 6 упоминается vectorizer.pkl. 
    # Вероятно, под vectorizer понимается процесс извлечения эмбеддингов.
    # Чтобы следовать букве задания, я создам пустой или фиктивный объект, 
    # если это потребуется, но лучше адаптирую код инференса.
    
    # 12. Сохранение в файл
    with open("day_4/baseline_results.txt", "w") as f:
        f.write("Baseline Results (Logistic Regression on CLS embeddings)\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Macro F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print("\nРезультаты сохранены в day_4/baseline_results.txt")

if __name__ == "__main__":
    run_baseline()
