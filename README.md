# Sentiment Classifier

Fine-tune **DistilBERT** for binary sentiment classification using **LoRA** (Low-Rank Adaptation), then use it to classify any text as positive or negative.

---

**[Try the live demo on HuggingFace Spaces](https://huggingface.co/spaces/EricRistol/sentiment-classifier)**


## Why this project

Large Language Models are powerful out of the box, but they're not great at domain-specific tasks without fine-tuning. Full fine-tuning of even a small model like DistilBERT (66M parameters) means updating every weight — expensive and easy to overfit on small datasets.

LoRA solves this by freezing the base weights and injecting tiny trainable matrices into the attention layers. Only ~0.3% of parameters are trainable, the adapter file is ~1 MB, and training finishes in minutes on a laptop CPU.

This is the same technique (at smaller scale) that companies use to customize GPT and LLaMA for production.

---

**[Try the live demo on HuggingFace Spaces](https://huggingface.co/spaces/EricRistol/sentiment-classifier)**


## What's in the repo

```
llm-fine-tuner/
├── data.py              synthetic sentiment dataset generator + tokenizer
├── train.py             base vs LoRA vs full fine-tune comparison
├── predict.py           load adapter, classify any text interactively
├── main.py              CLI menu (I - VII)
├── test_pipeline.py     pytest tests (8 tests)
├── requirements.txt
├── api/
│   ├── app.py           FastAPI server (loads model once, serves predictions)
│   └── static/
│       └── index.html   web demo (type a review, see the sentiment live)
├── data/                generated train/val/test CSVs
├── models/              LoRA adapter + comparison metrics
└── plots/               training loss curve
```

## Models compared

| Model                        | What it shows                                      |
|------------------------------|----------------------------------------------------|
| **Base DistilBERT (zero-shot)** | Baseline — no training, just pretrained weights. |
| **DistilBERT + LoRA**        | Parameter-efficient: only 0.3% of weights updated. |
| **DistilBERT full fine-tune** | All parameters updated — the upper bound.          |

Metrics: accuracy and F1. The interesting result is that LoRA matches full fine-tuning while training 300x fewer parameters.

## LoRA details

| Parameter            | Value               |
|----------------------|---------------------|
| Base model           | distilbert-base-uncased (66M params) |
| Target modules       | q_lin, v_lin        |
| LoRA rank (r)        | 8                   |
| LoRA alpha           | 16                  |
| Trainable params     | ~0.3% of total      |
| Adapter size on disk | ~1 MB               |

## How to run

```bash
pip install -r requirements.txt
python main.py         # interactive menu
```

Or drive each piece directly:

```bash
python data.py          # generate synthetic dataset
python train.py         # train all three models, save comparison
python predict.py       # classify sentences interactively
pytest -q               # run the tests
```

NOTE: First run downloads DistilBERT (~260 MB) from HuggingFace. After that it's cached.

## Web demo (API)

After training, launch the API server:

```bash
python main.py          # pick option VI
# or directly:
uvicorn api.app:app --reload
```

Then open **http://localhost:8000** in your browser. You get a clean page where you type a movie review and see the sentiment prediction with a confidence bar. The model loads once at startup so responses are fast.

Endpoints:
- `GET /` — the web demo
- `POST /predict` — send `{"text": "your review"}`, get back label + confidence + probabilities
- `GET /health` — server health check

## What I'd do next

- Swap the synthetic data for the full **IMDB dataset** (25k reviews) and compare performance.
- Try **DistilGPT-2** for text generation fine-tuning (generate reviews in a given style).
- Experiment with different LoRA ranks (4, 16, 32) and plot accuracy vs adapter size.
- Add **QLoRA** (quantized LoRA) to fine-tune a 7B model on consumer hardware.
- ~~Deploy the classifier as a **FastAPI endpoint** with the adapter loaded at startup.~~ Done!
- Deploy the API to a cloud service (Render, Railway, or HuggingFace Spaces).

## Tests

```bash
pytest -q
```

Covers: CSV generation, label balance, tokenization shapes, training produces artifacts, LoRA beats base, predictions return valid labels, and sentiment direction is correct.

---

**[Try the live demo on HuggingFace Spaces](https://huggingface.co/spaces/EricRistol/sentiment-classifier)**


Built as part of my AI/ML portfolio. Feedback and issues welcome.
