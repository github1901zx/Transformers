from transformers import AutoTokenizer

def explain_tokenization(text, model_name="bert-base-uncased"):
    """
    Демонстрирует и объясняет процесс токенизации для заданного текста.
    """
    print(f"--- Объяснение токенизации для модели: {model_name} ---")
    print(f"Исходный текст: {text}\n")
    
    # 1. Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Базовая токенизация (разбиение на токены)
    tokens = tokenizer.tokenize(text)
    print(f"1. Токены (разбиение): {tokens}")
    
    # 3. Преобразование в ID (input_ids)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"2. ID токенов: {token_ids}")
    
    # 4. Полная подготовка (с добавлением специальных токенов)
    encoded_input = tokenizer(text)
    full_input_ids = encoded_input['input_ids']
    print(f"3. Полные Input IDs (со спец. токенами): {full_input_ids}")
    
    # 5. Декодирование каждого ID для наглядности
    print("\nДетальный разбор:")
    for idx in full_input_ids:
        decoded = tokenizer.decode([idx])
        print(f"  ID {idx:6}  =>  '{decoded}'")
        
    print("\nСпециальные токены, используемые этой моделью:")
    print(f"  CLS (начало): {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"  SEP (разделитель): {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"  PAD (заполнение): {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  UNK (неизвестный): {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"  MASK (маскирование): {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")

if __name__ == "__main__":
    example_text = "Transformers are amazing and versatile!"
    explain_tokenization(example_text)
    
    print("\n" + "="*50 + "\n")
    
    example_text_ru = "Трансформеры — это мощная архитектура нейросетей."
    explain_tokenization(example_text_ru, model_name="bert-base-multilingual-cased")
