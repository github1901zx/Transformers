import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

def get_attention_matrices(text, model_name="bert-base-uncased"):
    print(f"\n--- Извлечение матриц внимания для модели: {model_name} ---")
    
    # 1. Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # output_attentions=True позволяет извлекать веса внимания
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    
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

def visualize_all_heads(tokens, attentions, layer_idx, save_path):
    """
    Визуализирует все головы внимания одного слоя на одной фигуре.
    """
    num_heads = attentions[layer_idx].shape[1]
    cols = 4
    rows = math.ceil(num_heads / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = axes.flatten()

    for head_idx in range(num_heads):
        matrix = attentions[layer_idx][0][head_idx].numpy()
        sns.heatmap(
            matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="YlGnBu",
            cbar=False,
            ax=axes[head_idx],
        )
        axes[head_idx].set_title(f"Head {head_idx + 1}")
        axes[head_idx].tick_params(axis="x", rotation=45)
        axes[head_idx].tick_params(axis="y", rotation=0)

    for idx in range(num_heads, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"All Attention Heads — Layer {layer_idx + 1}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Карты всех голов слоя {layer_idx + 1} сохранены в '{save_path}'")
    plt.close()

def analyze_attention_heads(tokens, attentions, layer_idx=11, target_word="transformers"):
    """
    Анализирует, на что смотрят разные головы на указанном слое.
    """
    print(f"\n--- Анализ голов внимания на слое {layer_idx+1} (токен: '{target_word}') ---")
    num_heads = attentions[layer_idx].shape[1]
    
    if target_word not in tokens:
        print(f"Токен '{target_word}' не найден в последовательности: {tokens}")
        return

    idx = tokens.index(target_word)
    for head in range(num_heads):
        matrix = attentions[layer_idx][0][head]
        max_attention_idx = torch.argmax(matrix[idx]).item()
        print(
            f"Head {head+1}: Токен '{target_word}' больше всего смотрит на "
            f"'{tokens[max_attention_idx]}' (вес: {matrix[idx][max_attention_idx]:.4f})"
        )

if __name__ == "__main__":
    os.makedirs("day_3", exist_ok=True)

    # Базовый пример
    text = "Transformers are fast and they are powerful."
    tokens, attentions = get_attention_matrices(text)
    num_layers = len(attentions)
    middle_layer = num_layers // 2 - 1  # 6-й слой (1-indexed) для BERT-base
    middle_layer = max(0, middle_layer)

    visualize_attention(
        tokens, attentions, layer_idx=0, head_idx=0,
        save_path="day_3/attention_layer1_head1.png",
    )
    visualize_attention(
        tokens, attentions, layer_idx=middle_layer, head_idx=0,
        save_path="day_3/attention_layer6_head1.png",
    )
    visualize_attention(
        tokens, attentions, layer_idx=11, head_idx=5,
        save_path="day_3/attention_layer12_head6.png",
    )
    visualize_all_heads(
        tokens, attentions, layer_idx=middle_layer,
        save_path="day_3/attention_layer6_all_heads.png",
    )
    analyze_attention_heads(tokens, attentions, layer_idx=11, target_word="transformers")

    # Анализ текста с sentiment-словом
    sentiment_text = "This film was terrible and the acting was awful."
    sentiment_tokens, sentiment_attentions = get_attention_matrices(sentiment_text)

    visualize_attention(
        sentiment_tokens, sentiment_attentions, layer_idx=middle_layer, head_idx=0,
        save_path="day_3/attention_sentiment_layer6_head1.png",
    )
    visualize_all_heads(
        sentiment_tokens, sentiment_attentions, layer_idx=11,
        save_path="day_3/attention_sentiment_layer12_all_heads.png",
    )
    analyze_attention_heads(
        sentiment_tokens, sentiment_attentions, layer_idx=11, target_word="terrible",
    )
