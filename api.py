from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

# Загрузка модели и токенизатора
MODEL_PATH = './fine_tuned_model'
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

class PredictionRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: PredictionRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    # В нашем случае 0 - Negative, 1 - Positive (исходя из Day 4/5)
    # В описании задачи Day 7 упомянут Neutral, но наш датасет был бинарным.
    # Оставим label_map гибким.
    label_map = {0: 'Negative', 1: 'Positive'}

    return {
        "text": request.text,
        "prediction": label_map.get(pred, str(pred)),
        "probabilities": {
            label_map.get(i, str(i)): float(probs[0][i])
            for i in range(len(probs[0]))
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
