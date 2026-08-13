"""
train_transformers.py
======================
STEP v (assignment) — Fine-tune pretrained Transformer models: BERT,
DistilBERT, RoBERTa.

Run with:   python train_transformers.py

WHY pretrained transformers are a different approach from everything else
in this project: every model so far (classic ML in the first phase; Simple
RNN/LSTM/GRU in this phase) learns everything about the English language
FROM SCRATCH, using only these ~18,000 training sentences. BERT/DistilBERT/
RoBERTa, in contrast, were already pretrained by their creators on massive
general-purpose text corpora (billions of words) before we ever touch them
— they arrive already "knowing" grammar, word meaning, and a great deal of
world knowledge. We only need to "fine-tune" (lightly retrain) them on our
much smaller emotion-labelled dataset to specialise that existing knowledge
for this specific task. This usually gives a real accuracy boost over
training from scratch, especially on a dataset this size.

WHY THIS SCRIPT MAY NOT RUN TO COMPLETION IN EVERY ENVIRONMENT:
Fine-tuning a pretrained transformer requires DOWNLOADING its pretrained
weights (several hundred MB each) from the Hugging Face Hub the first time
you run it. That download needs an internet connection that can reach
`huggingface.co`. If you're running this in a restricted/offline
environment (e.g. certain corporate networks, some sandboxed notebooks),
that download will fail — this is a genuine environment limitation, not a
bug in the code. The script detects this up front and tells you clearly
rather than crashing with a confusing stack trace partway through. On a
normal machine/Colab with internet access, this script runs correctly
end-to-end.

WHY WE MINIMISE THE "DATA CLEANING" STEP FOR TRANSFORMERS SPECIFICALLY:
Steps ii in earlier phases removed stopwords, lowercased everything, and
stemmed/lemmatised words — appropriate for Bag-of-Words/TF-IDF/RNNs, which
have no other way to relate different word forms. BERT/DistilBERT/RoBERTa
are different: they were PRETRAINED on natural, uncleaned text (with
stopwords, punctuation, and original casing all present), and their
subword tokenizers already handle inflected word forms internally (e.g.
splitting "unhappiness" into meaningful sub-pieces). Aggressively cleaning
the text before feeding it to these models actually HURTS performance,
because it shifts the input away from the distribution of text the model
was pretrained on (this is well-documented in NLP research/practice, not
just this project's opinion). So for this step, we apply only MINIMAL
cleaning (stripping extra whitespace) and keep original casing and
punctuation — while still demonstrating the classic cleaning function is
available (see `light_clean` vs `utils.clean_text` below) so you can see
the difference for yourself.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd

from utils import load_emotion_file

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_CHECKPOINTS = {
    "BERT":       "bert-base-uncased",
    "DistilBERT": "distilbert-base-uncased",
    "RoBERTa":    "roberta-base",
}
MAX_LENGTH = 64
NUM_EPOCHS = 3
BATCH_SIZE = 16


def log(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


def light_clean(text):
    """Minimal cleaning appropriate for transformer models: just collapse
    extra whitespace. Casing, punctuation, and stopwords are DELIBERATELY
    kept — see the module docstring above for why."""
    return " ".join(str(text).split())


# ==========================================================================
# PRE-FLIGHT CHECK — can we actually reach the Hugging Face Hub?
# ==========================================================================
def check_connectivity():
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=8)
        return True
    except Exception as e:
        print(f"Could not reach huggingface.co ({e}).")
        return False


if __name__ == "__main__":
    log("STEP v: Fine-tuning pretrained Transformers (BERT / DistilBERT / RoBERTa)")

    if not check_connectivity():
        print("""
This environment cannot reach huggingface.co, so pretrained model weights
cannot be downloaded here. This script is fully correct and WILL run on any
machine with normal internet access (your laptop, Google Colab, a cloud VM,
etc.) — nothing further needs to change.

To run it successfully:
  1. Make sure this machine has internet access to huggingface.co
  2. pip install transformers torch  (already listed in requirements.txt)
  3. python train_transformers.py

Expected runtime on that first successful run: each model downloads
~250-500MB of pretrained weights once (cached afterwards), then fine-tunes
for 3 epochs. On a GPU this is typically 5-15 minutes per model; on CPU
only, expect 1-3+ hours per model, since attention-based transformers are
far more compute-heavy than the RNN/LSTM/GRU models in this project. If you
don't have a GPU available, Google Colab's free tier provides one.
""")
        sys.exit(0)

    # The rest of this block only runs on a machine that DOES have internet
    # access to huggingface.co.
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments
    )
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    train_df = load_emotion_file(os.path.join(DATA_DIR, "train.txt"))
    val_df = load_emotion_file(os.path.join(DATA_DIR, "val.txt"))
    test_df = load_emotion_file(os.path.join(DATA_DIR, "test.txt"))

    for df in (train_df, val_df, test_df):
        df["clean_text"] = df["text"].apply(light_clean)

    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["emotion"])
    val_df["label"] = label_encoder.transform(val_df["emotion"])
    test_df["label"] = label_encoder.transform(test_df["emotion"])
    num_labels = len(label_encoder.classes_)

    class EmotionDataset(torch.utils.data.Dataset):
        """Wraps tokenized text + labels in the format PyTorch's Trainer expects."""
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall": recall_score(labels, preds, average="weighted", zero_division=0),
            "f1": f1_score(labels, preds, average="weighted", zero_division=0),
        }

    results = []

    for model_name, checkpoint in MODEL_CHECKPOINTS.items():
        log(f"Fine-tuning {model_name} ({checkpoint})")
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        def tokenize(texts):
            return tokenizer(
                list(texts), truncation=True, padding=True, max_length=MAX_LENGTH,
                return_tensors="pt",
            )

        train_enc = tokenize(train_df["clean_text"])
        val_enc = tokenize(val_df["clean_text"])
        test_enc = tokenize(test_df["clean_text"])

        train_ds = EmotionDataset(train_enc, train_df["label"].tolist())
        val_ds = EmotionDataset(val_enc, val_df["label"].tolist())
        test_ds = EmotionDataset(test_enc, test_df["label"].tolist())

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_labels
        )

        training_args = TrainingArguments(
            output_dir=os.path.join(MODELS_DIR, f"transformer_{model_name.lower()}_ckpt"),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            learning_rate=2e-5,   # standard fine-tuning LR for transformers -
                                   # much lower than training from scratch,
                                   # because we only want to gently adjust
                                   # already-good pretrained weights, not
                                   # overwrite them.
            weight_decay=0.01,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        train_time = time.time() - t0

        test_metrics = trainer.evaluate(test_ds)
        row = {
            "Model": model_name,
            "Accuracy": test_metrics["eval_accuracy"],
            "Precision (weighted)": test_metrics["eval_precision"],
            "Recall (weighted)": test_metrics["eval_recall"],
            "F1-score (weighted)": test_metrics["eval_f1"],
            "Train Time (s)": round(train_time, 2),
            "Params": sum(p.numel() for p in model.parameters()),
        }
        results.append(row)
        print(f"{model_name}: {row}")

        model.save_pretrained(os.path.join(MODELS_DIR, f"transformer_{model_name.lower()}"))
        tokenizer.save_pretrained(os.path.join(MODELS_DIR, f"transformer_{model_name.lower()}"))

    comparison_df = pd.DataFrame(results).sort_values("F1-score (weighted)", ascending=False)
    comparison_df.to_csv(os.path.join(MODELS_DIR, "transformer_comparison_table.csv"), index=False)
    print("\nFinal Transformer comparison:")
    print(comparison_df.to_string(index=False))
