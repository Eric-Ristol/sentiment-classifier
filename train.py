#Trains a handful of sentiment classifiers, compares them, saves the best one.

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, classification_report

import data

MODEL_NAME = "distilbert-base-uncased"
MODELS_DIR = "models"
PLOTS_DIR = "plots"

#Hyperparameters -- small so it finishes in <2 min on a laptop CPU.
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-4
MAX_LENGTH = 128

#LoRA config. rank=8 is a sweet spot for this task size.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

def make_dataloader(encodings, batch_size=BATCH_SIZE, shuffle=False):
    #Wraps tokenized data into a DataLoader. No custom class needed.
    ds = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        encodings["labels"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def evaluate(model, loader, device):
    #Runs the model on a dataloader and returns accuracy + f1.
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, mask, labels in loader:
            out = model(input_ids=input_ids.to(device), attention_mask=mask.to(device))
            preds = torch.argmax(out.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    return {"accuracy": acc, "f1": f1, "labels": all_labels, "preds": all_preds}

def train_loop(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR):
    #Standard training loop. Returns the list of per-step losses.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for input_ids, mask, labels in train_loader:
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=mask.to(device),
                labels=labels.to(device),
            )
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += out.loss.item()
            losses.append(out.loss.item())

        avg = epoch_loss / len(train_loader)
        val = evaluate(model, val_loader, device)
        print("    Epoch " + str(epoch + 1) + "/" + str(epochs) +
              "  loss=" + str(round(avg, 4)) +
              "  val_acc=" + str(round(val["accuracy"], 4)) +
              "  val_f1=" + str(round(val["f1"], 4)))
    return losses

def plot_loss(losses, save_path):
    #Training loss per step. A downward curve = the model is learning.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss (LoRA fine-tuning)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)

def run_training():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Device: " + str(device))

    print(">>> Preparing data...")
    train_df, val_df, test_df = data.load_splits()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = data.tokenize_df(tokenizer, train_df, MAX_LENGTH)
    val_enc = data.tokenize_df(tokenizer, val_df, MAX_LENGTH)
    test_enc = data.tokenize_df(tokenizer, test_df, MAX_LENGTH)
    train_loader = make_dataloader(train_enc, shuffle=True)
    val_loader = make_dataloader(val_enc)
    test_loader = make_dataloader(test_enc)
    print("    train=" + str(len(train_df)) + " val=" + str(len(val_df)) +
          " test=" + str(len(test_df)))

    results = []

    print("\n>>> Evaluating base DistilBERT (zero-shot)...")
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    base_model.to(device)
    base_metrics = evaluate(base_model, test_loader, device)
    results.append({"name": "Base DistilBERT (zero-shot)",
                     "accuracy": base_metrics["accuracy"], "f1": base_metrics["f1"]})
    print("    accuracy=" + str(round(base_metrics["accuracy"], 4)) +
          "  f1=" + str(round(base_metrics["f1"], 4)))

    print("\n>>> LoRA fine-tuning (r=" + str(LORA_R) + ", alpha=" + str(LORA_ALPHA) + ")...")
    lora_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    lora_model.to(device)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_lin", "v_lin"],
    )
    lora_model = get_peft_model(lora_model, lora_config)
    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())
    print("    Trainable: " + str(trainable) + " / " + str(total) +
          " (" + str(round(100 * trainable / total, 2)) + "%)")
    lora_losses = train_loop(lora_model, train_loader, val_loader, device)
    lora_metrics = evaluate(lora_model, test_loader, device)
    results.append({"name": "DistilBERT + LoRA",
                     "accuracy": lora_metrics["accuracy"], "f1": lora_metrics["f1"]})

    print("\n>>> Full fine-tuning (all parameters)...")
    full_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    full_model.to(device)
    trainable_full = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print("    Trainable: " + str(trainable_full) + " (100%)")
    train_loop(full_model, train_loader, val_loader, device)
    full_metrics = evaluate(full_model, test_loader, device)
    results.append({"name": "DistilBERT full fine-tune",
                     "accuracy": full_metrics["accuracy"], "f1": full_metrics["f1"]})

    #Save LoRA adapter (the interesting artifact -- just ~1 MB).
    adapter_dir = os.path.join(MODELS_DIR, "lora_adapter")
    lora_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    #Save the full model too so predict.py can use whichever is available.
    joblib.dump("distilbert-base-uncased", os.path.join(MODELS_DIR, "base_model_name.joblib"))

    #Comparison CSV.
    comp = pd.DataFrame(results)[["name", "accuracy", "f1"]]
    comp.to_csv(os.path.join(MODELS_DIR, "comparison.csv"), index=False)

    #Classification report for the LoRA model.
    report = classification_report(
        lora_metrics["labels"], lora_metrics["preds"],
        target_names=["negative", "positive"],
    )
    with open(os.path.join(MODELS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    #Training loss plot.
    plot_loss(lora_losses, os.path.join(PLOTS_DIR, "training_loss.png"))

    print("\n=== Sorted by accuracy ===")
    print(comp.sort_values("accuracy", ascending=False).to_string(index=False))

    best = comp.sort_values("accuracy", ascending=False).iloc[0]
    print("\nBest: " + best["name"] +
          " (accuracy=" + str(round(best["accuracy"], 4)) + ")")
    print("LoRA adapter saved to: " + adapter_dir)
    return comp

if __name__ == "__main__":
    run_training()
