#Pytest tests. Covers data generation, training, and prediction.

import os
import pandas as pd

import data
import train
import predict

def test_generate_creates_csvs():
    data.generate_dataset(n_train=40, n_val=10, n_test=10, seed=99)
    for path in [data.TRAIN_CSV, data.VAL_CSV, data.TEST_CSV]:
        assert os.path.exists(path)
    df = pd.read_csv(data.TRAIN_CSV)
    assert set(df.columns) == {"text", "label"}
    assert len(df) == 40

def test_generate_default_size():
    #Regenerate with default size for later tests.
    data.generate_dataset()
    df = pd.read_csv(data.TRAIN_CSV)
    assert len(df) == 800

def test_labels_are_balanced():
    df = pd.read_csv(data.TRAIN_CSV)
    pos_frac = df["label"].mean()
    assert 0.35 < pos_frac < 0.65

def test_tokenization_shapes():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(train.MODEL_NAME)
    df = pd.read_csv(data.TRAIN_CSV)
    enc = data.tokenize_df(tokenizer, df, max_length=64)
    assert enc["input_ids"].shape[0] == 800
    assert enc["input_ids"].shape[1] == 64
    assert enc["attention_mask"].shape == enc["input_ids"].shape
    assert len(enc["labels"]) == 800

def test_training_produces_artifacts():
    train.run_training()
    assert os.path.exists(os.path.join("models", "comparison.csv"))
    assert os.path.exists(os.path.join("models", "lora_adapter"))
    assert os.path.exists(os.path.join("plots", "training_loss.png"))

def test_lora_beats_base():
    comp = pd.read_csv(os.path.join("models", "comparison.csv"))
    base_acc = comp[comp["name"].str.contains("zero-shot")]["accuracy"].iloc[0]
    lora_acc = comp[comp["name"].str.contains("LoRA")]["accuracy"].iloc[0]
    assert lora_acc > base_acc

def test_predict_returns_valid_label():
    tokenizer, model = predict.load_model()
    result = predict.classify("This movie was amazing!", tokenizer, model)
    assert result["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(result["prob_negative"] + result["prob_positive"] - 1.0) < 0.01

def test_predict_sentiment_direction():
    tokenizer, model = predict.load_model()
    pos = predict.classify("Absolutely wonderful, best movie ever!", tokenizer, model)
    neg = predict.classify("Terrible and boring, worst film I have seen.", tokenizer, model)
    assert pos["label"] == "POSITIVE"
    assert neg["label"] == "NEGATIVE"
