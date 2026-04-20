#Everything related to the data: generating synthetic movie reviews,

import os
import random
import pandas as pd

DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

POSITIVE_TEMPLATES = [
    "I loved this movie, it was absolutely {adj}.",
    "An {adj} film with great performances all around.",
    "One of the best movies I have seen this year, truly {adj}.",
    "This film was {adj}, I would recommend it to anyone.",
    "A {adj} experience from start to finish, highly recommended.",
    "The acting was {adj} and the story kept me hooked.",
    "What a {adj} movie, the director did an amazing job.",
    "Everything about this film was {adj}, I enjoyed every minute.",
    "This was a {adj} story with memorable characters.",
    "Brilliant direction and {adj} cinematography make this a must-see.",
]

POSITIVE_ADJS = [
    "fantastic", "wonderful", "brilliant", "amazing", "outstanding",
    "excellent", "superb", "incredible", "magnificent", "great",
    "delightful", "captivating", "riveting", "spectacular", "impressive",
]

NEGATIVE_TEMPLATES = [
    "I hated this movie, it was completely {adj}.",
    "A {adj} film with no redeeming qualities.",
    "One of the worst movies I have ever seen, truly {adj}.",
    "This film was {adj}, I would not recommend it.",
    "A {adj} waste of time from beginning to end.",
    "The acting was {adj} and the story made no sense.",
    "What a {adj} movie, the director should be embarrassed.",
    "Everything about this film was {adj}, I regret watching it.",
    "This was a {adj} story with forgettable characters.",
    "Poor direction and {adj} writing make this one to avoid.",
]

NEGATIVE_ADJS = [
    "terrible", "awful", "dreadful", "horrible", "atrocious",
    "abysmal", "pathetic", "mediocre", "boring", "disappointing",
    "painful", "unbearable", "laughable", "miserable", "dismal",
]

def generate_dataset(n_train=800, n_val=100, n_test=100, seed=42):
    #Generates train/val/test CSVs with synthetic movie reviews.
    #Each row has "text" and "label" (0=negative, 1=positive).
    rng = random.Random(seed)
    os.makedirs(DATA_DIR, exist_ok=True)

    def _make_samples(n):
        rows = []
        for _ in range(n):
            if rng.random() < 0.5:
                tmpl = rng.choice(POSITIVE_TEMPLATES)
                adj = rng.choice(POSITIVE_ADJS)
                rows.append({"text": tmpl.format(adj=adj), "label": 1})
            else:
                tmpl = rng.choice(NEGATIVE_TEMPLATES)
                adj = rng.choice(NEGATIVE_ADJS)
                rows.append({"text": tmpl.format(adj=adj), "label": 0})
        return pd.DataFrame(rows)

    train_df = _make_samples(n_train)
    val_df = _make_samples(n_val)
    test_df = _make_samples(n_test)

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print("Generated " + str(n_train) + " train / " +
          str(n_val) + " val / " + str(n_test) + " test samples.")
    return train_df, val_df, test_df

def load_splits():
    #Loads the CSVs. Auto-generates if they don't exist yet.
    if not os.path.exists(TRAIN_CSV):
        generate_dataset()
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    return train_df, val_df, test_df

def tokenize_df(tokenizer, df, max_length=128):
    #Tokenizes a dataframe's "text" column. Returns a dict with
    #input_ids, attention_mask, and labels as tensors.
    import torch
    enc = tokenizer(
        df["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc["labels"] = torch.tensor(df["label"].values, dtype=torch.long)
    return enc

if __name__ == "__main__":
    generate_dataset()
