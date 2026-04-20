#FastAPI wrapper around the sentiment classifier.

import os
import sys

#Add the project root to the path so we can import predict.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import predict

app = FastAPI(
    title="Sentiment Classifier API",
    description="DistilBERT + LoRA fine-tuned for movie review sentiment.",
    version="1.0.0",
)

#Serve the static folder (CSS, JS, images if we ever add them).
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

#Load the model once when the server starts, not on every request.
tokenizer = None
model = None

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
    prob_negative: float
    prob_positive: float

@app.on_event("startup")
def load_model_on_startup():
    global tokenizer, model
    print("Loading LoRA model...")
    tokenizer, model = predict.load_model()
    print("Model ready.")

@app.get("/")
def serve_frontend():
    #Serves the main HTML page.
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(req: PredictRequest):
    #Classifies a piece of text and returns the result.
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    result = predict.classify(req.text, tokenizer, model)
    return result

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
