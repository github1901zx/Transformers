import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

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

def prepare_data():
    data = {
        'text': [
            "I love this movie, it's fantastic!",
            "Great acting and wonderful plot.",
            "Best film I've seen this year.",
            "Amazing experience, highly recommended.",
            "I really enjoyed this cinema masterpiece.",
            "Terrible movie, waste of time.",
            "I hated the plot and the acting was bad.",
            "Worst film ever. Don't watch it.",
            "Boring and predictable story.",
            "I didn't like it at all, very disappointing."
        ] * 20, # 200 примеров
        'label': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 20
    }
    df = pd.DataFrame(data)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label']
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

def main():
    # Настройки
    model_name = "distilbert-base-uncased"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"Using device: {device}")
    
    # 1. Загрузка токенизатора и данных
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_texts, val_texts, train_labels, val_labels = prepare_data()
    
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # 2. Загрузка модели
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # 3. Оптимизатор
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # ЗАДАЧА 7: Обучение модели
    num_epochs = 3
    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Accuracy: {val_acc:.4f}')
        print(f'Val F1: {val_f1:.4f}')
        print('-' * 50)

    # ЗАДАЧА 8: Сохранение модели
    print("\nSaving model and metrics...")
    save_path = './fine_tuned_model'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Сохранение метрик
    with open('fine_tuned_results.txt', 'w') as f:
        f.write(f'Final Validation F1: {val_f1:.4f}\n')
        f.write(f'Final Validation Accuracy: {val_acc:.4f}\n')
    
    print(f"Model saved to {save_path}")
    print("Metrics saved to fine_tuned_results.txt")

if __name__ == "__main__":
    main()
