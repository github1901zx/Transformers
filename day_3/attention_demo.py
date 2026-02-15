import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_attention_matrices(text, model_name="bert-base-uncased"):
    print(f"\n--- Извлечение матриц внимания для модели: {model_name} ---")
    
    # 1. Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # output_attentions=True позволяет извлекать веса внимания
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    
    # 2. Подготовка входных данных
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # 3. Получение выхода модели
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Attentions - кортеж из 12 тензоров (по количеству слоев)
    # Каждый тензор: [batch_size, num_heads, seq_len, seq_len]
    attentions = outputs.attentions
    
    print(f"Текст: {text}")
    print(f"Количество слоев с вниманием: {len(attentions)}")
    print(f"Размерность тензора внимания на одном слое: {attentions[0].shape}")
    
    return tokens, attentions

def visualize_attention(tokens, attentions, layer_idx=0, head_idx=0, save_path="day_3/attention_heatmap.png"):
    """
    Визуализирует матрицу внимания для конкретного слоя и головы.
    """
    # Извлекаем матрицу для нужного слоя и головы [seq_len, seq_len]
    attention_matrix = attentions[layer_idx][0][head_idx].numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, 
                annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    
    plt.title(f"Attention Map (Layer {layer_idx+1}, Head {head_idx+1})")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Тепловая карта внимания сохранена в '{save_path}'")
    plt.close()

def analyze_attention_heads(tokens, attentions, layer_idx=11):
    """
    Анализирует, на что смотрят разные головы на последнем слое.
    """
    print(f"\n--- Анализ голов внимания на слое {layer_idx+1} ---")
    num_heads = attentions[layer_idx].shape[1]
    
    for head in range(min(4, num_heads)): # Посмотрим на первые 4 головы
        matrix = attentions[layer_idx][0][head]
        # Найдем для каждого токена тот, на который он больше всего "смотрит" (исключая самого себя)
        # Для простоты просто выведем макс. связь для токена "transformers" если он есть
        
        target_word = "transformers"
        if target_word in tokens:
            idx = tokens.index(target_word)
            max_attention_idx = torch.argmax(matrix[idx]).item()
            print(f"Head {head+1}: Токен '{target_word}' больше всего смотрит на '{tokens[max_attention_idx]}' (вес: {matrix[idx][max_attention_idx]:.4f})")

if __name__ == "__main__":
    # Создаем папку если её нет
    os.makedirs("day_3", exist_ok=True)
    
    text = "Transformers are fast and they are powerful."
    tokens, attentions = get_attention_matrices(text)
    
    # Визуализируем первую голову первого слоя (часто сфокусирована на соседних токенах или [CLS])
    visualize_attention(tokens, attentions, layer_idx=0, head_idx=0, save_path="day_3/attention_layer1_head1.png")
    
    # Визуализируем одну из голов последнего слоя (часто более семантически нагружена)
    visualize_attention(tokens, attentions, layer_idx=11, head_idx=5, save_path="day_3/attention_layer12_head6.png")
    
    analyze_attention_heads(tokens, attentions)
