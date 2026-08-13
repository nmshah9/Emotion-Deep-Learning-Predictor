"""
utils_dl.py
-----------
Shared utilities for the Deep Learning / Transformer phase of the Emotion
Detection project. Builds on the same text-cleaning function used in the
classic-ML phase (`clean_text`, imported from `utils.py`) so both phases
clean text identically and results are comparable.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import clean_text, load_emotion_file  # noqa: F401  (re-exported for convenience)


def build_tokenizer(texts, num_words=10000, oov_token="<OOV>"):
    """
    Fits a Keras Tokenizer on the TRAINING texts only.

    WHY fit on training data only: the tokenizer builds a word->integer
    vocabulary from whatever text it sees. If we fit it on validation/test
    text too, the model would implicitly "see" words from data it's
    supposed to be evaluated on (data leakage), making the reported
    performance overly optimistic and not representative of real-world use
    on brand-new sentences.
    """
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded(texts, tokenizer, maxlen=40):
    """
    Converts a list/Series of cleaned text strings into a fixed-size 2-D
    array of integers ready to feed into an Embedding layer.

    WHY padding is needed: neural network layers expect every input in a
    batch to have the SAME shape, but sentences naturally have different
    lengths. `pad_sequences` solves this by cutting longer sentences down
    to `maxlen` tokens and padding shorter ones with zeros, so every row
    ends up exactly `maxlen` long.

    WHY padding='post' (zeros added at the END, not the start): with
    'post' padding, the real words always start at position 0, which
    keeps the embedding lookup for the actual content consistent
    regardless of how much padding follows. (Note: for basic
    RNN/LSTM/GRU layers as used in this project, either pre- or
    post-padding works reasonably; 'post' is simply a common, easy-to-
    reason-about default here.)

    WHY maxlen=40: covers roughly the 95th percentile of sentence length
    in this dataset (checked during development) — long enough to rarely
    truncate real content, short enough to keep training fast.
    """
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")


def plot_training_history(history, model_name, save_dir="plots"):
    """
    Plots training vs validation accuracy AND loss side by side for one
    trained model, and saves the figure to disk.

    WHY this plot is the key overfitting/underfitting diagnostic:
      - If TRAIN accuracy keeps climbing while VALIDATION accuracy
        plateaus or drops (equivalently: train loss keeps falling while
        val loss climbs) -> the model is memorising the training data
        instead of learning generalisable patterns = OVERFITTING.
      - If BOTH train and validation accuracy stay low / both losses stay
        high and flat -> the model isn't learning the patterns in the data
        at all (too simple an architecture, too few epochs, or too high a
        learning rate) = UNDERFITTING.
      - If train and validation curves track closely together and both
        reach a good accuracy -> that's the well-fit case we want.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title(f"{model_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title(f"{model_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("/", "-")
    path = os.path.join(save_dir, f"{safe_name}_history.png")
    plt.savefig(path, dpi=100)
    plt.close(fig)
    return path


def diagnose_fit(history):
    """
    Turns the raw training-history numbers into a short, human-readable
    verdict: "Overfitting", "Underfitting", or "Good fit" - plus the gap
    size, so the comparison table can show this at a glance instead of
    requiring everyone to read every plot by eye.

    Heuristic used (simple and transparent on purpose):
      - final_train_acc - final_val_acc > 0.08  -> "Overfitting"
      - final_train_acc < 0.55 (both train & val accuracy low)
                                                -> "Underfitting"
      - otherwise                              -> "Good fit"
    These thresholds are reasonable defaults for this 6-class problem, not
    universal constants - always sanity-check against the actual plot too.
    """
    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    gap = train_acc - val_acc

    if train_acc < 0.55 and val_acc < 0.55:
        verdict = "Underfitting"
    elif gap > 0.08:
        verdict = "Overfitting"
    else:
        verdict = "Good fit"

    return {
        "final_train_acc": round(float(train_acc), 4),
        "final_val_acc": round(float(val_acc), 4),
        "train_val_gap": round(float(gap), 4),
        "verdict": verdict,
    }
