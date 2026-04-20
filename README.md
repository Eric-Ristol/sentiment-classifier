# Sentiment Classifier

Fine-tune DistilBERT for sentiment classification using LoRA.

**[Live demo](https://huggingface.co/spaces/EricRistol/sentiment-classifier)**

## What it does

LoRA (Low-Rank Adaptation) lets you fine-tune a model without changing all the weights. Instead of updating all 66M parameters in DistilBERT, LoRA adds tiny trainable matrices (~0.3% of parameters) into the attention layers. Only these get updated during training.

The adapter file ends up being ~1 MB instead of 260 MB, and training takes minutes instead of hours.

## What's in here

```
├── data.py              generate synthetic dataset + tokenizer
├── train.py             compare base model vs LoRA vs full fine-tune
├── predict.py           load adapter and classify text
├── main.py              CLI menu (options I-VII)
├── test_pipeline.py     pytest tests
├── api/
│   ├── app.py           FastAPI server with live predictions
│   └── static/
│       └── index.html   web UI
├── data/                training/val/test CSVs
├── models/              LoRA adapter + metrics
└── plots/               training curves
```

## Results

Tested on a small synthetic dataset (positive/negative reviews):

| Model | Accuracy | F1 |
|-------|----------|-----|
| Base model | 0.72 | 0.70 |
| LoRA fine-tune | 0.94 | 0.93 |
| Full fine-tune | 0.95 | 0.94 |

LoRA gets almost the same accuracy as full fine-tuning but with 300x fewer parameters trained.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

Or run individual scripts:

```bash
python data.py              # generate dataset
python train.py             # train models
python predict.py           # classify a sentence
pytest -q                   # run tests
```

First run downloads DistilBERT (~260 MB) from HuggingFace. It's cached after that.

## Web demo

```bash
python main.py              # pick option VI
# or:
uvicorn api.app:app --reload
```

Open http://localhost:8000. Type a movie review and see the sentiment with a confidence score.

**API endpoints:**
- `GET /` — the web demo
- `POST /predict` — send `{"text": "review"}`, get label + confidence
- `GET /health` — health check

## Tests

```bash
pytest -q
```

Checks CSV generation, label balance, tokenization, training artifacts, that LoRA beats base, predictions return valid labels, and sentiment is correct.

---

**[Live demo](https://huggingface.co/spaces/EricRistol/sentiment-classifier)**
