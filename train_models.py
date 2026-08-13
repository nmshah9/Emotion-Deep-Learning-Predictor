"""
train_dl_models.py
===================
DEEP LEARNING PHASE — Emotion Detection

Run with:   python train_dl_models.py

Covers assignment steps i, ii, iii, vi (partially vii):
    i.   Load the dataset
    ii.  Clean the text (same cleaning as the classic-ML phase)
    iii. Train Simple RNN, LSTM, GRU, Bidirectional RNN/LSTM/GRU, and
         Stacked RNN/LSTM/GRU architectures
    vi.  Diagnose overfitting/underfitting for every model via train-vs-
         validation curves
    vii. Save a comparison table (this script's part of it — KerasTuner
         and transformer results are appended by the other two scripts)

WHY DEEP LEARNING AT ALL, on top of the classic ML models we already built:
Classic ML models (Logistic Regression, Decision Tree, Random Forest) treat
each sentence as an unordered "bag" of word counts — they have no notion of
word ORDER. "I was not happy, I was sad" and "I was happy, I was not sad"
would look identical to a Bag-of-Words model. Recurrent architectures
(RNN/LSTM/GRU) process a sentence word-by-word IN ORDER, carrying forward a
"memory" of what came before, which lets them pick up on sequences, simple
negation patterns, and word order that bag-of-words approaches structurally
cannot see.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, SimpleRNN, LSTM, GRU, Bidirectional, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from utils_dl import build_tokenizer, texts_to_padded, plot_training_history, diagnose_fit
from utils import clean_text, load_emotion_file

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

VOCAB_SIZE = 10000     # only keep the 10,000 most frequent words - rare/typo
                        # words contribute little and bloat the embedding table
EMBED_DIM = 64          # size of each word's learned vector representation
MAXLEN = 40             # sentence length after padding/truncation (~95th percentile)
EPOCHS = 10             # capped, with EarlyStopping to stop sooner if it converges
BATCH_SIZE = 128
RANDOM_STATE = 42


def log(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


# ==========================================================================
# STEP i & ii — LOAD + CLEAN
# ==========================================================================
log("STEP 1-2: Loading & cleaning data")

train_df = load_emotion_file(os.path.join(DATA_DIR, "train.txt"))
val_df = load_emotion_file(os.path.join(DATA_DIR, "val.txt"))
test_df = load_emotion_file(os.path.join(DATA_DIR, "test.txt"))

# For deep learning we keep the ORIGINAL train/val split (rather than
# merging them like the classic-ML phase did) because Keras' .fit() wants
# an explicit validation set at every epoch to plot the overfitting/
# underfitting curves this script produces — that's a different need than
# GridSearchCV's internal cross-validation from the ML phase.
for df in (train_df, val_df, test_df):
    df["clean_text"] = df["text"].apply(lambda x: clean_text(x, method="lemmatize"))

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

label_encoder = LabelEncoder()
y_train = to_categorical(label_encoder.fit_transform(train_df["emotion"]))
y_val = to_categorical(label_encoder.transform(val_df["emotion"]))
y_test_int = label_encoder.transform(test_df["emotion"])
y_test = to_categorical(y_test_int)
NUM_CLASSES = len(label_encoder.classes_)
print("Classes:", list(label_encoder.classes_))

# ==========================================================================
# TOKENIZATION — turn cleaned text into padded integer sequences
# ==========================================================================
# WHY a Tokenizer + Embedding layer instead of TF-IDF/BoW here: RNN-family
# models need to see WORDS IN ORDER, one at a time, to build up their
# internal "memory" of the sentence. TF-IDF/BoW collapse a sentence into a
# single fixed-size vector of counts with no order information, which is
# incompatible with how recurrent layers work. So each word is instead
# mapped to an integer ID (Tokenizer), and the Embedding layer learns a
# dense vector for each ID DURING training - the network itself learns
# which words behave similarly, rather than us hand-crafting that.
tokenizer = build_tokenizer(train_df["clean_text"], num_words=VOCAB_SIZE)
X_train = texts_to_padded(train_df["clean_text"], tokenizer, maxlen=MAXLEN)
X_val = texts_to_padded(val_df["clean_text"], tokenizer, maxlen=MAXLEN)
X_test = texts_to_padded(test_df["clean_text"], tokenizer, maxlen=MAXLEN)
print("Padded shape:", X_train.shape)

joblib.dump(tokenizer, os.path.join(MODELS_DIR, "dl_tokenizer.pkl"))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, "dl_label_encoder.pkl"))


# ==========================================================================
# STEP iii — MODEL ARCHITECTURES
# ==========================================================================
# WHY each variant is included, and what it adds over the previous one:
#
#   Simple RNN        - the most basic recurrent layer. Struggles with
#                        longer sentences due to the "vanishing gradient"
#                        problem: information from early words fades out by
#                        the time the network reaches later words. Included
#                        as a baseline to show what LSTM/GRU improve on.
#   LSTM               - adds gated "memory cells" specifically designed to
#                        preserve information over longer sequences, fixing
#                        SimpleRNN's vanishing-gradient weakness at the cost
#                        of more parameters (slower to train).
#   GRU                - a simplified gating mechanism vs LSTM (fewer gates,
#                        fewer parameters) that often reaches similar
#                        accuracy to LSTM while training faster.
#   Bidirectional      - reads the sentence BOTH forward and backward and
#   (LSTM/GRU)           combines both views. Helps because a word's emotional
#                        meaning can depend on words that come AFTER it, not
#                        just before ("I thought it was terrible, but I was
#                        wrong" - the sentiment-flipping word arrives late).
#   Stacked            - stacks two recurrent layers on top of each other,
#   (LSTM/GRU)           letting the second layer learn higher-level patterns
#                        from the first layer's output - more representational
#                        capacity, at the cost of being more prone to
#                        overfitting and slower to train.
#
# Every architecture below shares the same Embedding layer setup and
# Dense(softmax) output layer, so the ONLY thing that differs between
# models is the recurrent layer itself - keeping the comparison fair.

def build_model(rnn_type, bidirectional=False, stacked=False, units=64, dropout=0.3):
    layer_cls = {"simplernn": SimpleRNN, "lstm": LSTM, "gru": GRU}[rnn_type]

    model = Sequential()
    # mask_zero=True is critical here: our sentences are padded with 0s up
    # to MAXLEN, and roughly half of most sentences (median length 17 vs.
    # MAXLEN=40) is padding, not real words. Without masking, the recurrent
    # layer keeps processing those padding steps AFTER the real sentence
    # ends and its hidden state gets diluted/overwritten by meaningless
    # zero-embeddings, which in testing caused LSTM/GRU to get stuck
    # predicting only the majority class. mask_zero=True tells every
    # downstream recurrent layer to skip padded timesteps entirely.
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, mask_zero=True))

    if stacked:
        first_layer = layer_cls(units, return_sequences=True)
        second_layer = layer_cls(units)
        if bidirectional:
            model.add(Bidirectional(first_layer))
            model.add(Bidirectional(second_layer))
        else:
            model.add(first_layer)
            model.add(second_layer)
    else:
        layer = layer_cls(units)
        model.add(Bidirectional(layer) if bidirectional else layer)

    model.add(Dropout(dropout))   # randomly zeroes some activations during
                                    # training only - a standard regulariser
                                    # that reduces overfitting by preventing
                                    # the network from relying too heavily
                                    # on any single neuron.
    model.add(Dense(32, activation="relu"))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


MODEL_CONFIGS = [
    ("Simple RNN",              dict(rnn_type="simplernn", bidirectional=False, stacked=False)),
    ("LSTM",                    dict(rnn_type="lstm",       bidirectional=False, stacked=False)),
    ("GRU",                     dict(rnn_type="gru",        bidirectional=False, stacked=False)),
    ("Bidirectional LSTM",      dict(rnn_type="lstm",       bidirectional=True,  stacked=False)),
    ("Bidirectional GRU",       dict(rnn_type="gru",        bidirectional=True,  stacked=False)),
    ("Stacked LSTM",            dict(rnn_type="lstm",       bidirectional=False, stacked=True)),
    ("Stacked GRU",             dict(rnn_type="gru",        bidirectional=False, stacked=True)),
]

log(f"STEP 3: Training {len(MODEL_CONFIGS)} recurrent architectures")

results = []
histories = {}

# EarlyStopping: if validation loss stops improving for 2 straight epochs,
# stop training and roll back to the best-performing epoch's weights. This
# both saves time AND is itself an overfitting guard - it prevents us from
# continuing to train (and overfit) long after the model stopped actually
# improving on unseen data.
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

import sys
import json

RESULTS_JSON = os.path.join(MODELS_DIR, "dl_results_partial.json")

def load_partial_results():
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return {}

def save_partial_results(all_results):
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

partial_results = load_partial_results()

# Allow running a single model by name from the command line, e.g.:
#   python train_dl_models.py "Bidirectional LSTM"
# This lets training resume model-by-model if a run gets interrupted,
# instead of losing all progress and starting over from scratch.
only_model = sys.argv[1] if len(sys.argv) > 1 else None
configs_to_run = [c for c in MODEL_CONFIGS if only_model is None or c[0] == only_model]

for name, cfg in configs_to_run:
    if name in partial_results:
        print(f"Skipping {name} (already trained, found in {RESULTS_JSON})")
        results.append(partial_results[name])
        continue
    print(f"\n--- Training: {name} ---")
    t0 = time.time()
    model = build_model(**cfg)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )
    train_time = time.time() - t0
    histories[name] = history

    plot_path = plot_training_history(history, name, save_dir=PLOTS_DIR)
    fit_diag = diagnose_fit(history)

    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    row = {
        "Model": name,
        "Accuracy": float(accuracy_score(y_test_int, preds)),
        "Precision (weighted)": float(precision_score(y_test_int, preds, average="weighted", zero_division=0)),
        "Recall (weighted)": float(recall_score(y_test_int, preds, average="weighted", zero_division=0)),
        "F1-score (weighted)": float(f1_score(y_test_int, preds, average="weighted", zero_division=0)),
        "Epochs Run": int(len(history.history["loss"])),
        "Train Time (s)": round(float(train_time), 2),
        "Final Train Acc": fit_diag["final_train_acc"],
        "Final Val Acc": fit_diag["final_val_acc"],
        "Train-Val Gap": fit_diag["train_val_gap"],
        "Fit Diagnosis": fit_diag["verdict"],
        "Params": int(model.count_params()),
    }
    results.append(row)
    print(f"{name}: test_acc={row['Accuracy']:.4f}  f1={row['F1-score (weighted)']:.4f}  "
          f"diagnosis={row['Fit Diagnosis']}  time={train_time:.1f}s  plot={plot_path}")

    # Save every model so KerasTuner / the notebook can reload any of them
    # later without retraining from scratch.
    model.save(os.path.join(MODELS_DIR, f"dl_{name.replace(' ', '_').lower()}.keras"))

    # Persist progress after EVERY model (not just at the end), so a crash
    # or interruption never loses more than the model currently in flight.
    partial_results[name] = row
    save_partial_results(partial_results)

if only_model is not None:
    print(f"\nFinished single-model run for '{only_model}'. Re-run without an "
          f"argument once all models are done to print the full comparison table.")
    sys.exit(0)


# ==========================================================================
# COMPARISON TABLE (DL models only - combined with ML + Transformers later)
# ==========================================================================
log("Deep Learning model comparison")

dl_comparison_df = pd.DataFrame(results).sort_values("F1-score (weighted)", ascending=False).reset_index(drop=True)
print(dl_comparison_df.to_string(index=False))
dl_comparison_df.to_csv(os.path.join(MODELS_DIR, "dl_comparison_table.csv"), index=False)

best_dl_row = dl_comparison_df.iloc[0]
print(f"\n>>> BEST DEEP LEARNING MODEL (pre-tuning): {best_dl_row['Model']} "
      f"(F1-weighted = {best_dl_row['F1-score (weighted)']:.4f}, "
      f"fit = {best_dl_row['Fit Diagnosis']})")

print(f"\nSaved: {MODELS_DIR}/dl_comparison_table.csv")
print(f"Saved training-curve plots to: {PLOTS_DIR}/")
print("\nNext: run  python tune_dl_model.py   to hyperparameter-tune the best architecture with KerasTuner.")
