#Loads the fine-tuned LoRA adapter and classifies text as positive or negative.

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

ADAPTER_DIR = os.path.join("models", "lora_adapter")
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
LABEL_NAMES = {0: "NEGATIVE", 1: "POSITIVE"}

def load_model():
    #Loads base DistilBERT + the LoRA adapter on top. Raises if not trained yet.
    if not os.path.exists(ADAPTER_DIR):
        raise FileNotFoundError(
            "No adapter found at " + ADAPTER_DIR + ". Run `python train.py` first."
        )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    return tokenizer, model

def classify(text, tokenizer, model):
    #Returns a dict with label, confidence, and per-class probabilities.
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    probs = torch.softmax(out.logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs))
    return {
        "text": text,
        "label": LABEL_NAMES[pred_id],
        "confidence": round(float(probs[pred_id]), 4),
        "prob_negative": round(float(probs[0]), 4),
        "prob_positive": round(float(probs[1]), 4),
    }

def run_interactive():
    print("\n=== Sentiment classifier (DistilBERT + LoRA) ===")
    print("Type a sentence and press Enter. Type 'quit' to exit.\n")
    tokenizer, model = load_model()
    while True:
        text = input(">>> ").strip()
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break
        result = classify(text, tokenizer, model)
        print("  " + result["label"] +
              "  (confidence=" + str(result["confidence"]) +
              "  neg=" + str(result["prob_negative"]) +
              "  pos=" + str(result["prob_positive"]) + ")\n")

if __name__ == "__main__":
    run_interactive()
