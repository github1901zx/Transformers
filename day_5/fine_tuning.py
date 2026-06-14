import random
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentiment_data import load_sentiment_dataset
from model_config import MODEL_NAME, MODEL_REVISION, RANDOM_SEED
from model_artifacts import has_valid_weights, WEIGHTS_FILE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAVE_PATH = PROJECT_ROOT / "fine_tuned_model"
RESULTS_PATH = PROJECT_ROOT / "fine_tuned_results.txt"

# ==========================================
# ПОДГОТОВКА ДАННЫХ
# ==========================================

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    return device


def prepare_data():
    df = load_sentiment_dataset()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=RANDOM_SEED, stratify=df['label']
    )

    return train_texts, val_texts, train_labels, val_labels

# ==========================================
# ЗАДАЧА 6: Функции для обучения и оценки
# ==========================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')

    return accuracy, f1


def run_fine_tuning(save_path=None, num_epochs=3):
    set_seed(RANDOM_SEED)

    save_path = Path(save_path) if save_path is not None else DEFAULT_SAVE_PATH
    device = get_device()

    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME} (revision: {MODEL_REVISION})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    train_texts, val_texts, train_labels, val_labels = prepare_data()

    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, revision=MODEL_REVISION
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Accuracy: {val_acc:.4f}')
        print(f'Val F1: {val_f1:.4f}')
        print('-' * 50)

    print("\nSaving model and metrics...")
    save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    if not has_valid_weights():
        raise RuntimeError(
            f"Чекпоинт сохранён неполностью: отсутствует {WEIGHTS_FILE.name}. "
            f"Ожидается model.safetensors или pytorch_model.bin в {save_path}."
        )

    with open(RESULTS_PATH, 'w') as f:
        f.write(f'Model: {MODEL_NAME}\n')
        f.write(f'Revision: {MODEL_REVISION}\n')
        f.write(f'Random seed: {RANDOM_SEED}\n')
        f.write(f'Final Validation F1: {val_f1:.4f}\n')
        f.write(f'Final Validation Accuracy: {val_acc:.4f}\n')

    print(f"Model saved to {save_path}")
    print(f"Metrics saved to {RESULTS_PATH}")

    return {
        "save_path": save_path,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
    }


def main():
    run_fine_tuning()


if __name__ == "__main__":
    main()
