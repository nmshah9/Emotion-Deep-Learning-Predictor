"""
tune_dl_model.py
=================
STEP iv (assignment) — Hyperparameter tuning with KerasTuner.

Run with:   python tune_dl_model.py

WHY tune with KerasTuner instead of hand-picking numbers: choosing values
like "how many GRU units" or "what dropout rate" by hand is guesswork.
KerasTuner systematically trains and evaluates many different combinations
and keeps the one that scores best on a held-out validation set — the same
principle as GridSearchCV in the classic-ML phase, but adapted for neural
networks (where the "search space" includes architectural choices like
layer size, not just algorithm settings).

WHY we tune the GRU specifically: `train_dl_models.py` already trained and
compared seven architectures (Simple RNN, LSTM, GRU, their Bidirectional
and Stacked variants) and found plain GRU to be the best-performing AND
best-fitting (least overfit) of the untuned models. Tuning is expensive
(each trial fully trains a model), so rather than tuning all seven
architectures separately, we focus that budget on the strongest starting
point.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
TUNER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kt_tuning")
os.makedirs(MODELS_DIR, exist_ok=True)

VOCAB_SIZE = 10000
MAXLEN = 40


def log(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


# ==========================================================================
# LOAD + CLEAN + TOKENIZE (identical to train_dl_models.py, so results are
# directly comparable to the untuned GRU baseline)
# ==========================================================================
log("Loading & preparing data")

train_df = load_emotion_file(os.path.join(DATA_DIR, "train.txt"))
val_df = load_emotion_file(os.path.join(DATA_DIR, "val.txt"))
test_df = load_emotion_file(os.path.join(DATA_DIR, "test.txt"))
for df in (train_df, val_df, test_df):
    df["clean_text"] = df["text"].apply(lambda x: clean_text(x, method="lemmatize"))

label_encoder = LabelEncoder()
y_train = to_categorical(label_encoder.fit_transform(train_df["emotion"]))
y_val = to_categorical(label_encoder.transform(val_df["emotion"]))
y_test_int = label_encoder.transform(test_df["emotion"])
y_test = to_categorical(y_test_int)
NUM_CLASSES = len(label_encoder.classes_)

tokenizer = build_tokenizer(train_df["clean_text"], num_words=VOCAB_SIZE)
X_train = texts_to_padded(train_df["clean_text"], tokenizer, maxlen=MAXLEN)
X_val = texts_to_padded(val_df["clean_text"], tokenizer, maxlen=MAXLEN)
X_test = texts_to_padded(test_df["clean_text"], tokenizer, maxlen=MAXLEN)


# ==========================================================================
# SEARCH SPACE DEFINITION
# ==========================================================================
# WHY each hyperparameter is included in the search, and its range:
#   embed_dim   - size of each word's learned vector. Larger = more
#                 representational capacity but more parameters to learn
#                 (risk of overfitting on a dataset this size). Searched
#                 between 32 and 128.
#   gru_units   - size of the GRU's hidden state ("memory"). More units can
#                 capture more complex patterns, at higher compute/overfitting
#                 cost. Searched between 32 and 128.
#   dropout     - fraction of activations randomly zeroed during training,
#                 a direct regularisation control. Searched between 0.2 and
#                 0.5 — the untuned baseline used a fixed 0.3, so this lets
#                 the tuner check whether more or less regularisation helps.
#   learning_rate - step size for the Adam optimiser. Too high can prevent
#                 convergence (or, as we saw during development, cause the
#                 model to bounce out of a good solution); too low makes
#                 training painfully slow. Searched on a log scale between
#                 1e-4 and 1e-2, which is the standard range to consider.
def build_tunable_model(hp):
    embed_dim = hp.Int("embed_dim", min_value=32, max_value=128, step=32)
    gru_units = hp.Int("gru_units", min_value=32, max_value=128, step=32)
    dropout_rate = hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=embed_dim, mask_zero=True),
        GRU(gru_units),
        Dropout(dropout_rate),
        Dense(32, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ==========================================================================
# RUN THE SEARCH
# ==========================================================================
# WHY RandomSearch (rather than an exhaustive grid): our 4-hyperparameter
# space has 4 x 4 x 4 x (continuous log range) combinations - far too many
# to try exhaustively in reasonable time. RandomSearch samples a fixed
# number of random combinations (max_trials) instead, which in practice
# finds near-optimal configurations far faster than an exhaustive search,
# at the (accepted) cost of not guaranteeing the single best combination.
log("Running KerasTuner RandomSearch")

tuner = kt.RandomSearch(
    build_tunable_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    directory=TUNER_DIR,
    project_name="emotion_gru_tuning",
    overwrite=True,
    seed=42,
)

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

t0 = time.time()
tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=128,
    callbacks=[early_stop],
    verbose=2,
)
search_time = time.time() - t0

best_hp = tuner.get_best_hyperparameters(1)[0]
print("\nBest hyperparameters found:")
for name in ["embed_dim", "gru_units", "dropout", "learning_rate"]:
    print(f"  {name}: {best_hp.get(name)}")
print(f"Search took {search_time:.1f}s across 10 trials")


# ==========================================================================
# TRAIN THE FINAL MODEL WITH THE BEST HYPERPARAMETERS
# ==========================================================================
# WHY retrain rather than just reuse a trial's model: KerasTuner's trials
# use a shared, capped epoch budget to keep the search itself fast. Once we
# know the best hyperparameters, we retrain ONE final model with them,
# using EarlyStopping again to let this specific configuration train for as
# long as it actually keeps improving, rather than settling for whatever
# happened to be its state at the end of a tuning trial.
log("Training final model with best hyperparameters")

best_model = tuner.hypermodel.build(best_hp)
t0 = time.time()
history = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=128,
    callbacks=[early_stop],
    verbose=2,
)
train_time = time.time() - t0

plot_path = plot_training_history(history, "Tuned GRU (KerasTuner)", save_dir=PLOTS_DIR)
fit_diag = diagnose_fit(history)

preds = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
tuned_row = {
    "Model": "Tuned GRU (KerasTuner)",
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
    "Params": int(best_model.count_params()),
    "Best Hyperparameters": str({name: best_hp.get(name) for name in
                                  ["embed_dim", "gru_units", "dropout", "learning_rate"]}),
}

print(f"\nTuned GRU: test_acc={tuned_row['Accuracy']:.4f}  "
      f"f1={tuned_row['F1-score (weighted)']:.4f}  diagnosis={tuned_row['Fit Diagnosis']}")

# --------------------------------------------------------------------------
# Compare against the untuned baseline GRU so the value of tuning is explicit
# --------------------------------------------------------------------------
baseline_path = os.path.join(MODELS_DIR, "dl_comparison_table.csv")
if os.path.exists(baseline_path):
    dl_df = pd.read_csv(baseline_path)
    baseline_gru = dl_df[dl_df["Model"] == "GRU"]
    if not baseline_gru.empty:
        base_f1 = baseline_gru.iloc[0]["F1-score (weighted)"]
        print(f"\nUntuned GRU F1-weighted : {base_f1:.4f}")
        print(f"Tuned GRU F1-weighted   : {tuned_row['F1-score (weighted)']:.4f}")
        print(f"Improvement             : {tuned_row['F1-score (weighted)'] - base_f1:+.4f}")

    # Append the tuned result as a new row and re-save the comparison table
    combined_df = pd.concat([dl_df, pd.DataFrame([tuned_row])], ignore_index=True)
    combined_df = combined_df.sort_values("F1-score (weighted)", ascending=False).reset_index(drop=True)
    combined_df.to_csv(baseline_path, index=False)
    print(f"\nUpdated {baseline_path} with the tuned result.")

# Save the tuned model + its own artifacts (in case it becomes the overall
# best model once combined with the classic-ML and transformer results)
best_model.save(os.path.join(MODELS_DIR, "dl_tuned_gru.keras"))
joblib.dump(tokenizer, os.path.join(MODELS_DIR, "dl_tokenizer.pkl"))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, "dl_label_encoder.pkl"))

print(f"\nSaved tuned model to {MODELS_DIR}/dl_tuned_gru.keras")
print(f"Saved training-curve plot to {plot_path}")
