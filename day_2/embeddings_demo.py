import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

def get_embeddings(text, model_name="bert-base-uncased"):
    print(f"\n--- Анализ эмбеддингов для модели: {model_name} ---")
    
    # 1. Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # output_hidden_states=True позволяет извлекать состояния всех слоев
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    
    # 2. Подготовка входных данных
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # 3. Получение выхода модели
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Last Hidden State: [batch_size, sequence_length, hidden_size]
    last_hidden_state = outputs.last_hidden_state
    
    # Hidden States из всех слоев (если output_hidden_states=True)
    # Это кортеж из (embedding_layer + 12 encoder layers)
    hidden_states = outputs.hidden_states
    
    print(f"Текст: {text}")
    print(f"Размерность последнего скрытого слоя: {last_hidden_state.shape}")
    print(f"Количество слоев (включая эмбеддинги): {len(hidden_states)}")
    
    return tokens, last_hidden_state, hidden_states

def cosine_similarity(v1, v2):
    v1 = v1.flatten()
    v2 = v2.flatten()
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))

def demo_similarity():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    sentences = [
        "The bank of the river is beautiful.",
        "I need to go to the bank to withdraw money.",
        "The river flows slowly."
    ]
    
    embeddings = []
    word_index = [] # Индекс слова "bank" или "river" для сравнения
    
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        with torch.no_grad():
            output = model(**inputs)
        
        last_hidden = output.last_hidden_state[0]
        embeddings.append(last_hidden)
        
        # Найдем индексы интересующих нас слов
        if "bank" in tokens:
            word_index.append(tokens.index("bank"))
        elif "river" in tokens:
            word_index.append(tokens.index("river"))
        else:
            word_index.append(0)

    print("\n--- Сравнение контекстных эмбеддингов ---")
    # Сравним "bank" в реке и "bank" в финансах
    sim_12 = cosine_similarity(embeddings[0][word_index[0]], embeddings[1][word_index[1]])
    print(f"Сходство 'bank' (река) и 'bank' (деньги): {sim_12:.4f}")
    
    # Сравним "bank" в реке и "river" в реке
    sim_13 = cosine_similarity(embeddings[0][word_index[0]], embeddings[2][word_index[2]])
    print(f"Сходство 'bank' (река) и 'river' (река): {sim_13:.4f}")

def visualize_hidden_states(all_hidden, tokens):
    """
    Визуализирует 'активность' эмбеддингов для каждого слоя.
    """
    # Выбираем первый токен предложения для каждого из 13 слоев
    # all_hidden - кортеж из 13 тензоров [1, seq_len, 768]
    
    layer_means = [torch.mean(torch.abs(layer[0])).item() for layer in all_hidden]
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(layer_means)), layer_means, marker='o')
    plt.title("Средняя абсолютная активация по слоям")
    plt.xlabel("Номер слоя (0 - Embedding Layer)")
    plt.ylabel("Среднее значение активации")
    plt.grid(True)
    plt.savefig("day_2/layer_activations.png")
    print("\nГрафик активаций слоев сохранен в 'day_2/layer_activations.png'")

if __name__ == "__main__":
    text = "Transformers are powerful."
    tokens, last_hidden, all_hidden = get_embeddings(text)
    
    print("\nПример векторов для токенов (первые 5 значений):")
    for i, token in enumerate(tokens):
        vec_start = last_hidden[0][i][:5].tolist()
        formatted_vec = ", ".join([f"{v:.4f}" for v in vec_start])
        print(f"Token: {token:12} | Vec: [{formatted_vec}, ...]")
        
    visualize_hidden_states(all_hidden, tokens)
    demo_similarity()
