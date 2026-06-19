import sys
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentiment_data import load_sentiment_dataset, get_target_names
from model_artifacts import (
    require_fine_tuned_model,
    require_baseline_artifacts,
    load_baseline_artifacts,
    FINE_TUNED_MODEL_DIR,
)
from model_config import RANDOM_SEED

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    return device


def load_models(device):
    """Загружает ранее сохранённые fine-tuned и baseline модели."""
    require_fine_tuned_model()
    require_baseline_artifacts()

    model_ft = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer_ft = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)
    model_ft.eval()
    model_ft.to(device)

    baseline_model, baseline_feature_extractor, baseline_tokenizer, _ = load_baseline_artifacts(device)

    return model_ft, tokenizer_ft, baseline_model, baseline_feature_extractor, baseline_tokenizer


def predict_fine_tuned(texts, model, tokenizer, device):
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


def predict_baseline(texts, model, feature_extractor, tokenizer, device, clean_func=None):
    if isinstance(texts, str):
        texts = [texts]

    if clean_func:
        texts = [clean_func(t) for t in texts]

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = feature_extractor(**inputs)

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


def compare_examples(model_ft, tokenizer_ft, baseline_model, baseline_feature_extractor,
                     baseline_tokenizer, device):
    print("\n--- Сравнение на тестовых примерах ---")
    test_texts = [
        "This movie was absolutely fantastic!",
        "Terrible, waste of my time.",
        "It was okay, nothing special.",
        "Best film I've seen this year!",
        "Boring and too long."
    ]

    preds_ft = predict_fine_tuned(test_texts, model_ft, tokenizer_ft, device)
    preds_baseline = predict_baseline(
        test_texts, baseline_model, baseline_feature_extractor, baseline_tokenizer, device
    )

    for i, text in enumerate(test_texts):
        print(f'\nТекст: {text}')
        print(f'Fine-tuned: {preds_ft[i]["prediction"]} (probs: {preds_ft[i]["probabilities"]})')
        print(f'Baseline: {preds_baseline[i]["prediction"]}')
        print(f'Совпадают: {preds_ft[i]["prediction"] == preds_baseline[i]["prediction"]}')


def build_confusion_matrices(model_ft, tokenizer_ft, baseline_model, baseline_feature_extractor,
                             baseline_tokenizer, device):
    print("\n--- Построение Confusion Matrix ---")

    df = load_sentiment_dataset()
    _, test_df, _, _ = train_test_split(
        df, df['label'], test_size=0.2, random_state=RANDOM_SEED, stratify=df['label']
    )

    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    target_names = get_target_names(test_labels)

    print("Получение предсказаний для тестового набора...")
    preds_ft_all = predict_fine_tuned(test_texts, model_ft, tokenizer_ft, device)
    y_pred_ft = [p['prediction'] for p in preds_ft_all]

    preds_baseline_all = predict_baseline(
        test_texts, baseline_model, baseline_feature_extractor, baseline_tokenizer, device
    )
    y_pred_baseline = [p['prediction'] for p in preds_baseline_all]

    cm_ft = confusion_matrix(test_labels, y_pred_ft)
    cm_baseline = confusion_matrix(test_labels, y_pred_baseline)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        cm_ft, annot=True, fmt='d', ax=ax1, cmap='Blues',
        xticklabels=target_names, yticklabels=target_names,
    )
    ax1.set_title('Fine-tuned Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    sns.heatmap(
        cm_baseline, annot=True, fmt='d', ax=ax2, cmap='Greens',
        xticklabels=target_names, yticklabels=target_names,
    )
    ax2.set_title('Baseline Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    plt.tight_layout()
    output_path = PROJECT_ROOT / 'day_6' / 'confusion_matrices.png'
    plt.savefig(output_path)
    print(f"Матрицы ошибок сохранены в {output_path}")

    acc_ft = accuracy_score(test_labels, y_pred_ft)
    f1_ft = f1_score(test_labels, y_pred_ft, average='macro')
    acc_baseline = accuracy_score(test_labels, y_pred_baseline)
    f1_baseline = f1_score(test_labels, y_pred_baseline, average='macro')

    report_ft = classification_report(test_labels, y_pred_ft, target_names=target_names)
    report_baseline = classification_report(
        test_labels, y_pred_baseline, target_names=target_names
    )

    print(f"\nFine-tuned Accuracy: {acc_ft:.4f}")
    print(f"Fine-tuned F1 (macro): {f1_ft:.4f}")
    print("\nFine-tuned classification report:")
    print(report_ft)

    print(f"\nBaseline Accuracy: {acc_baseline:.4f}")
    print(f"Baseline F1 (macro): {f1_baseline:.4f}")
    print("\nBaseline classification report:")
    print(report_baseline)

    comparison_path = PROJECT_ROOT / 'day_6' / 'comparison_results.txt'
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


def main():
    print("Загрузка сохранённых моделей...")
    device = get_device()
    model_ft, tokenizer_ft, baseline_model, baseline_feature_extractor, baseline_tokenizer = load_models(device)

    compare_examples(
        model_ft, tokenizer_ft, baseline_model, baseline_feature_extractor, baseline_tokenizer, device
    )
    build_confusion_matrices(
        model_ft, tokenizer_ft, baseline_model, baseline_feature_extractor, baseline_tokenizer, device
    )


if __name__ == "__main__":
    main()
