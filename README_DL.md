# 🧠 Emotion Detection — Deep Learning & Transformers (Phase 2)

This is the **second phase** of the Emotion Detection project. Phase 1
(`emo_proj/` — the "Classic ML" project) used Logistic Regression, Decision
Tree, and Random Forest on Bag-of-Words/TF-IDF/Word2Vec features. This
phase asks a different question: **does a neural network that reads the
sentence as an ordered sequence do better?** — and goes one step further by
also trying pretrained Transformer models (BERT, DistilBERT, RoBERTa).

## What's inside this folder

```
emotion_detection_dl_project/
├── data/                        # same train/val/test.txt as phase 1
├── notebooks/
│   └── Emotion_Detection_DL_Transformers.ipynb
├── models/                      # created as you run the scripts below
│   ├── dl_*.keras                 (7 trained RNN/LSTM/GRU models)
│   ├── dl_tuned_gru.keras         (KerasTuner-tuned best model)
│   ├── dl_tokenizer.pkl / dl_label_encoder.pkl
│   ├── dl_comparison_table.csv
│   ├── transformer_*/             (only created if train_transformers.py
│   │                                completes successfully - see below)
│   └── FINAL_master_comparison_table.csv
├── plots/                       # train-vs-validation curve PNGs, one per model
├── utils.py                     # shared text cleaning (same as phase 1)
├── utils_dl.py                  # DL-specific helpers (tokenizing, plotting, fit diagnosis)
├── train_dl_models.py           # trains all 7 RNN/LSTM/GRU variants
├── tune_dl_model.py              # KerasTuner search on the best architecture
├── train_transformers.py          # fine-tunes BERT / DistilBERT / RoBERTa
├── build_final_comparison.py       # combines phase 1 + phase 2 results into one table
├── app_dl.py                         # Streamlit app - predicts emotion using the best DL model
├── requirements_dl.txt
└── README_DL.md                     # this file
```

> **This folder depends on phase 1's `utils.py`** (already copied in here)
> and, for the very final combined comparison table, on phase 1's
> `emo_proj/models/comparison_table.csv`. Keep both project folders
> side-by-side (as they were delivered) so `build_final_comparison.py` can
> find phase 1's results automatically.

---

## Setup (do this once)

Following on from phase 1's setup (same virtual environment, Python 3.12):

```bash
pip install -r requirements_dl.txt
```

This adds TensorFlow, KerasTuner, and Hugging Face `transformers` + `torch`
on top of phase 1's dependencies. **This is a substantially bigger install
than phase 1** (~1-2GB, mostly TensorFlow and PyTorch) — make sure you have
disk space and a decent internet connection for this one-time install.

---

## Running the project, in order

### Step 1 — Train the RNN/LSTM/GRU family

```bash
python train_dl_models.py
```

Trains Simple RNN, LSTM, GRU, Bidirectional LSTM/GRU, and Stacked LSTM/GRU
— 7 architectures total — each printing its progress, saving a trained
model file, and saving a train-vs-validation curve plot to `plots/`.
**Expected runtime: roughly 5-15 minutes** on a modern CPU (faster with a
GPU).

If you want to (re)train just ONE architecture (useful if a run gets
interrupted), pass its name as an argument, e.g.:
```bash
python train_dl_models.py "Bidirectional LSTM"
```
Progress is saved incrementally to `models/dl_results_partial.json` as each
model finishes, so re-running without an argument will skip any
already-completed models and just reprint the full comparison table.

**Reference results** (yours may vary slightly by hardware/library
version):

| Model | Accuracy | F1 (weighted) | Fit Diagnosis |
|---|---|---|---|
| GRU | 0.915 | 0.915 | Good fit |
| Stacked GRU | 0.901 | 0.902 | Good fit |
| Stacked LSTM | 0.900 | 0.901 | Overfitting |
| LSTM | 0.889 | 0.889 | Overfitting |
| Bidirectional LSTM | 0.886 | 0.887 | Overfitting |
| Bidirectional GRU | 0.886 | 0.887 | Overfitting |
| Simple RNN | 0.832 | 0.831 | Overfitting |

A plain **GRU** was both the best-performing AND the best-fit model —
somewhat counter to the "bigger/more complex is better" instinct. The
Bidirectional and Stacked variants add real parameters and training time
without a matching accuracy gain on a dataset this size (~18,000
sentences), and overfit more as a result.

### Step 2 — Hyperparameter-tune the best architecture

```bash
python tune_dl_model.py
```

Runs a KerasTuner `RandomSearch` (10 trials) over GRU's embedding size,
hidden units, dropout rate, and learning rate, then retrains one final
model with the winning combination. Appends the tuned result to
`models/dl_comparison_table.csv`. **Expected runtime: roughly 10-15
minutes.**

**Reference result:** Tuned GRU reached **0.919 F1-weighted** (embed_dim=128,
gru_units=64, dropout=0.3, learning_rate≈0.004) — a modest but real
improvement (+0.004) over the untuned GRU, still diagnosed as a "Good fit."

### Step 3 — Fine-tune pretrained Transformers (BERT / DistilBERT / RoBERTa)

```bash
python train_transformers.py
```

**Read this before running:** this step downloads pretrained model weights
(~250-500MB each, first run only) from the Hugging Face Hub, which needs
internet access to `huggingface.co`. If your network can't reach it (some
corporate/sandboxed networks block it), the script will print a clear
explanation and exit cleanly rather than crash — this is a genuine
environment limitation, not a bug.

**On a machine/notebook with internet access**, this fine-tunes all three
models for 3 epochs each with standard settings (learning rate 2e-5, the
conventional choice for lightly adapting pretrained weights rather than
overwriting them). **Expected runtime:**
- **With a GPU:** roughly 5-15 minutes per model (15-45 min total)
- **CPU only:** expect 1-3+ hours per model — transformers are far more
  compute-heavy than the RNN/LSTM/GRU models above, since they process
  every pair of words in a sentence against each other (self-attention)
  rather than one word at a time. **If you don't have a GPU, Google
  Colab's free tier provides one** — upload this project folder there for
  a much faster run.

Unlike the earlier cleaning steps, this script applies only *minimal*
text cleaning (whitespace normalisation, original casing/punctuation kept)
— see the detailed explanation in the script's docstring for why heavy
cleaning (stopword removal, stemming) can actually hurt pretrained
transformer performance.

### Step 4 — Build the final combined comparison table

```bash
python build_final_comparison.py
```

Pulls together phase 1's classic-ML results, this phase's Deep Learning
results, and (if Step 3 completed) the Transformer results into one ranked
table (`models/FINAL_master_comparison_table.csv`), and prints the overall
best model.

### Step 5 — Launch the Streamlit app

Once Step 1 (and ideally Step 2) has completed, so `models/dl_tokenizer.pkl`
and at least one `models/dl_*.keras` file exist:
```bash
streamlit run app_dl.py
```
Opens a browser at `http://localhost:8501` (or the next free port). Type a
sentence and click **Predict Emotion** to see the predicted emotion, a
confidence bar chart across all 6 emotions, and (by expanding a section) how
your text was cleaned before prediction. The app automatically loads the
best available model — the KerasTuner-tuned GRU if Step 2 has run, falling
back to the best untuned architecture from Step 1 otherwise — and the
sidebar shows which one is currently loaded plus the full comparison table.

To stop the app, go back to the terminal and press `Ctrl+C`.

---

## Overfitting / Underfitting — how to read it

Every trained model gets a `Fit Diagnosis` in the comparison tables and a
matching plot in `plots/`:

- **Overfitting**: training accuracy notably higher than validation
  accuracy (gap > 8 percentage points here) — the model is starting to
  memorise training examples rather than generalising. Look for training
  loss still falling while validation loss climbs.
- **Underfitting**: both training and validation accuracy stay low — the
  model isn't learning the patterns in the data. Look for both curves
  staying flat and high (loss) / flat and low (accuracy).
- **Good fit**: training and validation curves track closely and both
  reach solid accuracy.

Open any PNG in `plots/` to see this directly — the automatic diagnosis is
a convenient summary, but the shape of the curve is the real evidence.

---

## A bug we found and fixed during development (worth knowing about)

Early versions of the RNN/LSTM/GRU training got LSTM and GRU stuck
predicting only the majority class (~35% accuracy, completely flat
across epochs). The cause: sentences are padded with zeros up to 40 tokens,
but the median sentence is only ~17 words — so roughly half of most rows is
padding, not real content. Without telling the `Embedding` layer to mask
padding (`mask_zero=True`), the recurrent layers kept processing that
trailing padding *after* the real sentence ended, overwriting their hidden
state with noise. Adding `mask_zero=True` fixed it immediately — LSTM/GRU
accuracy jumped from ~35% to ~85-91%. This fix is already applied
throughout `train_dl_models.py`, `tune_dl_model.py`, and the notebook — just
flagging it in case you build on this code and hit the same symptom
elsewhere (a model stuck at exactly the majority-class proportion, flat
across every epoch, is the telltale sign).

---

## Troubleshooting

**`AttributeError: 'str' object has no attribute 'fillna'` from `build_final_comparison.py`**
This was a real bug, now fixed: it happened when `dl_comparison_table.csv`
didn't yet have a "Best Hyperparameters" column (i.e. `tune_dl_model.py`
hadn't been run yet, only `train_dl_models.py`). Update to the latest
`build_final_comparison.py` from this delivery — it now checks whether the
column exists before touching it, instead of assuming it's always there.

**`ModuleNotFoundError: No module named 'utils'` / `'utils_dl'`**
Run scripts from the project root (not from inside `notebooks/`). The
notebook fixes this automatically in its setup cell.

**`train_transformers.py` can't connect / times out**
See Step 3 above — this needs internet access to `huggingface.co`. Try
Google Colab if your usual machine/network can't reach it, or if you don't
have a GPU available.

**Training feels very slow**
Expected for the Transformer step on CPU-only hardware (see Step 3 runtime
notes). The RNN/LSTM/GRU steps (Steps 1-2) should be much faster (minutes,
not hours) even on CPU.

**Out of memory during KerasTuner search or Transformer fine-tuning**
Lower `BATCH_SIZE` in `tune_dl_model.py` / `train_transformers.py` (e.g.
from 128/16 down to 64/8) — smaller batches use less memory per step, at
the cost of slightly longer total training time.

**`build_final_comparison.py` can't find phase 1's results**
It looks for `../emo_proj/models/comparison_table.csv` relative to this
folder by default. If you've moved the phase 1 project elsewhere, edit the
`ML_MODELS_DIR` variable near the top of `build_final_comparison.py` to
point at wherever `comparison_table.csv` actually lives — or just skip it;
the script still produces a combined Deep-Learning-only table if phase 1's
results aren't found.
