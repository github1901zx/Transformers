from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from model_artifacts import ensure_fine_tuned_model, FINE_TUNED_MODEL_DIR

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    ensure_fine_tuned_model()
    model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)
    model.eval()
    yield


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(request: PredictionRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

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
