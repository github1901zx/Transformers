from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent

def get_embeddings(texts, tokenizer, model, batch_size=8):
    """
    Извлекает CLS-эмбеддинги для списка текстов батчами.
    Возвращает np.ndarray формы (n_texts, hidden_size).
    """
    model.eval()
    all_cls = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_cls.append(cls_vectors)

    return np.vstack(all_cls)

def analyze_single_text(text, model_name="bert-base-uncased"):
    """Демонстрация hidden states для одного текста (визуализация слоёв)."""
    print(f"\n--- Анализ эмбеддингов для модели: {model_name} ---")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    hidden_states = outputs.hidden_states

    print(f"Текст: {text}")
    print(f"Размерность последнего скрытого слоя: {last_hidden_state.shape}")
    print(f"Количество слоев (включая эмбеддинги): {len(hidden_states)}")

    return tokens, last_hidden_state, hidden_states

def cosine_similarity(v1, v2):
    v1 = v1.flatten()
    v2 = v2.flatten()
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))

def text_similarity(text1, text2, tokenizer, model):
    """
    Вычисляет косинусное сходство между двумя текстами через CLS-эмбеддинги.
    """
    embeddings = get_embeddings([text1, text2], tokenizer, model)
    v1 = torch.tensor(embeddings[0])
    v2 = torch.tensor(embeddings[1])
    return cosine_similarity(v1, v2).item()

def demo_text_similarity(tokenizer, model):
    """Демонстрация сходства целых текстов через get_embeddings."""
    pairs = [
        ("I love this movie, it's fantastic!", "Best film I've seen this year!"),
        ("I love this movie, it's fantastic!", "Terrible movie, waste of time."),
        ("The weather is sunny today.", "It is raining outside."),
    ]

    print("\n--- Сходство текстов (text_similarity) ---")
    for text_a, text_b in pairs:
        score = text_similarity(text_a, text_b, tokenizer, model)
        print(f"'{text_a}'")
        print(f"  vs '{text_b}'")
        print(f"  Cosine similarity: {score:.4f}\n")

def demo_similarity():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    sentences = [
        "The bank of the river is beautiful.",
        "I need to go to the bank to withdraw money.",
        "The river flows slowly.",
    ]

    embeddings = []
    word_index = []

    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            output = model(**inputs)

        last_hidden = output.last_hidden_state[0]
        embeddings.append(last_hidden)

        if "bank" in tokens:
            word_index.append(tokens.index("bank"))
        elif "river" in tokens:
            word_index.append(tokens.index("river"))
        else:
            word_index.append(0)

    print("\n--- Сравнение контекстных эмбеддингов ---")
    sim_12 = cosine_similarity(embeddings[0][word_index[0]], embeddings[1][word_index[1]])
    print(f"Сходство 'bank' (река) и 'bank' (деньги): {sim_12:.4f}")

    sim_13 = cosine_similarity(embeddings[0][word_index[0]], embeddings[2][word_index[2]])
    print(f"Сходство 'bank' (река) и 'river' (река): {sim_13:.4f}")

def visualize_hidden_states(all_hidden, tokens):
    layer_means = [torch.mean(torch.abs(layer[0])).item() for layer in all_hidden]

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(layer_means)), layer_means, marker="o")
    plt.title("Средняя абсолютная активация по слоям")
    plt.xlabel("Номер слоя (0 - Embedding Layer)")
    plt.ylabel("Среднее значение активации")
    plt.grid(True)
    output_path = OUTPUT_DIR / "layer_activations.png"
    plt.savefig(output_path)
    print(f"\nГрафик активаций слоев сохранен в '{output_path}'")

if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    texts = [
        "Transformers are powerful.",
        "BERT uses contextual embeddings.",
    ]
    cls_embeddings = get_embeddings(texts, tokenizer, model, batch_size=8)
    print("--- CLS-эмбеддинги (get_embeddings) ---")
    print(f"Форма: {cls_embeddings.shape}")
    print(f"Первые 5 значений [0]: {cls_embeddings[0][:5]}\n")

    text = "Transformers are powerful."
    tokens, last_hidden, all_hidden = analyze_single_text(text)

    print("\nПример векторов для токенов (первые 5 значений):")
    for i, token in enumerate(tokens):
        vec_start = last_hidden[0][i][:5].tolist()
        formatted_vec = ", ".join([f"{v:.4f}" for v in vec_start])
        print(f"Token: {token:12} | Vec: [{formatted_vec}, ...]")

    visualize_hidden_states(all_hidden, tokens)
    demo_text_similarity(tokenizer, model)
    demo_similarity()
