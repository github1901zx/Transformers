import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentiment_data import get_labeled_data

# ==========================================
# ЗАДАЧА 1: Загрузка моделей
# ==========================================

print("Загрузка моделей...")

# 1. Загрузка fine-tuned модели
model_ft = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_model')
tokenizer_ft = AutoTokenizer.from_pretrained('./fine_tuned_model')
model_ft.eval()

# Перемещение на доступное устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
model_ft.to(device)

# 2. Загрузка baseline модели
# В нашей реализации baseline использует DistilBERT как экстрактор эмбеддингов
baseline_model = joblib.load('baseline_model.pkl')
# В качестве "vectorizer" для baseline мы используем базовый DistilBERT
baseline_feature_extractor = AutoModel.from_pretrained("distilbert-base-uncased")
baseline_feature_extractor.eval()
baseline_feature_extractor.to(device)
baseline_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ==========================================
# ЗАДАЧА 2: Функция предсказания для fine-tuned
# ==========================================

def predict_fine_tuned(texts, model, tokenizer):
    if isinstance(texts, str):
        texts = [texts]

    predictions = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        predictions.append({
            'text': text,
            'prediction': pred,
            'probabilities': probs[0].cpu().numpy()
        })

    return predictions

# ==========================================
# ЗАДАЧА 3: Функция предсказания для baseline
# ==========================================

def predict_baseline(texts, model, feature_extractor, tokenizer, clean_func=None):
    if isinstance(texts, str):
        texts = [texts]

    if clean_func:
        texts = [clean_func(t) for t in texts]

    # Извлечение эмбеддингов (аналог vectorizer.transform)
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = feature_extractor(**inputs)
        
        # CLS токен
        cls_output = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_output)
    
    X = np.vstack(embeddings)
    predictions = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

    results = []
    for i, text in enumerate(texts):
        results.append({
            'text': text,
            'prediction': int(predictions[i]),
            'probabilities': probs[i] if probs is not None else None
        })

    return results

# ==========================================
# ЗАДАЧА 4: Сравнение на примерах
# ==========================================

print("\n--- Сравнение на тестовых примерах ---")
test_texts = [
    "This movie was absolutely fantastic!",
    "Terrible, waste of my time.",
    "It was okay, nothing special.",
    "Best film I've seen this year!",
    "Boring and too long."
]

# Fine-tuned predictions
preds_ft = predict_fine_tuned(test_texts, model_ft, tokenizer_ft)

# Baseline predictions
preds_baseline = predict_baseline(test_texts, baseline_model, baseline_feature_extractor, baseline_tokenizer)

for i, text in enumerate(test_texts):
    print(f'\nТекст: {text}')
    print(f'Fine-tuned: {preds_ft[i]["prediction"]} (probs: {preds_ft[i]["probabilities"]})')
    print(f'Baseline: {preds_baseline[i]["prediction"]}')
    print(f'Совпадают: {preds_ft[i]["prediction"] == preds_baseline[i]["prediction"]}')

# ==========================================
# ЗАДАЧА 5: Confusion Matrix для обеих моделей
# ==========================================

print("\n--- Построение Confusion Matrix ---")

# 1. Подготовка тестовых данных (тот же уникальный датасет, что и в Day 5)
df = pd.DataFrame(get_labeled_data())
_, test_df, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=42, stratify=df['label'])

test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

# 2. Получение предсказаний
print("Получение предсказаний для тестового набора...")
preds_ft_all = predict_fine_tuned(test_texts, model_ft, tokenizer_ft)
y_pred_ft = [p['prediction'] for p in preds_ft_all]

preds_baseline_all = predict_baseline(test_texts, baseline_model, baseline_feature_extractor, baseline_tokenizer)
y_pred_baseline = [p['prediction'] for p in preds_baseline_all]

# 3. Построение confusion matrix
cm_ft = confusion_matrix(test_labels, y_pred_ft)
cm_baseline = confusion_matrix(test_labels, y_pred_baseline)

# Визуализация
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_ft, annot=True, fmt='d', ax=ax1, cmap='Blues')
ax1.set_title('Fine-tuned Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')

sns.heatmap(cm_baseline, annot=True, fmt='d', ax=ax2, cmap='Greens')
ax2.set_title('Baseline Confusion Matrix')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')

plt.tight_layout()
plt.savefig('day_6/confusion_matrices.png')
print("Матрицы ошибок сохранены в day_6/confusion_matrices.png")

# Вывод итоговых метрик и classification_report
acc_ft = accuracy_score(test_labels, y_pred_ft)
f1_ft = f1_score(test_labels, y_pred_ft, average='macro')
acc_baseline = accuracy_score(test_labels, y_pred_baseline)
f1_baseline = f1_score(test_labels, y_pred_baseline, average='macro')

report_ft = classification_report(test_labels, y_pred_ft, target_names=['Negative', 'Positive'])
report_baseline = classification_report(test_labels, y_pred_baseline, target_names=['Negative', 'Positive'])

print(f"\nFine-tuned Accuracy: {acc_ft:.4f}")
print(f"Fine-tuned F1 (macro): {f1_ft:.4f}")
print("\nFine-tuned classification report:")
print(report_ft)

print(f"\nBaseline Accuracy: {acc_baseline:.4f}")
print(f"Baseline F1 (macro): {f1_baseline:.4f}")
print("\nBaseline classification report:")
print(report_baseline)

comparison_path = 'day_6/comparison_results.txt'
with open(comparison_path, 'w', encoding='utf-8') as f:
    f.write("=== Сравнение Fine-tuned и Baseline ===\n\n")
    f.write(f"Fine-tuned Accuracy: {acc_ft:.4f}\n")
    f.write(f"Fine-tuned F1 (macro): {f1_ft:.4f}\n\n")
    f.write("Fine-tuned classification report:\n")
    f.write(report_ft)
    f.write("\n")
    f.write(f"Baseline Accuracy: {acc_baseline:.4f}\n")
    f.write(f"Baseline F1 (macro): {f1_baseline:.4f}\n\n")
    f.write("Baseline classification report:\n")
    f.write(report_baseline)

print(f"\nРезультаты сравнения сохранены в {comparison_path}")
