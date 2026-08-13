"""
app_dl.py
=========
STREAMLIT WEB APP for the Deep Learning phase of the Emotion Detection
project.

Run with:   streamlit run app_dl.py

WHAT THIS FILE DOES:
Loads the best-performing Keras model trained by train_models.py /
tune_model.py (by default, the KerasTuner-tuned GRU, since it scored
highest in testing — falls back to the best untuned model automatically if
the tuned one isn't present), along with the same tokenizer and label
encoder used during training. The user types a sentence, we clean it with
the SAME function used during training (from utils.py), convert it to a
padded token sequence with the SAME fitted tokenizer, and feed it to the
model to get a predicted emotion + confidence scores for every class.

IMPORTANT: This app does NOT train anything itself. You must run
`python train_models.py` (and, optionally, `python tune_model.py`
for the best result) at least once before `streamlit run app_dl.py` will
work — see README_DL.md.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from utils import clean_text
from utils_dl import texts_to_padded
from PIL import Image

st.set_page_config(page_title="Emotion Detector (Deep Learning)", page_icon="🧠", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MAXLEN = 40

EMOTION_EMOJI = {"joy": "😀", "sadness": "😢", "anger": "😠", "fear": "😨", "love": "❤️", "surprise": "😲"}
EMOTION_COLOR = {"joy": "#FFD93D", "sadness": "#4D96FF", "anger": "#FF4D4D",
                  "fear": "#9D4DFF", "love": "#FF6FA8", "surprise": "#4DFFB8"}

# Preference order: try the tuned model first (best result in testing),
# then fall back through the untuned architectures so the app still works
# even if you've only run train_dl_models.py and skipped tuning.
MODEL_PREFERENCE = [
    ("Tuned GRU (KerasTuner)", "dl_tuned_gru.keras"),
    ("GRU", "dl_gru.keras"),
    ("Stacked GRU", "dl_stacked_gru.keras"),
    ("Stacked LSTM", "dl_stacked_lstm.keras"),
    ("LSTM", "dl_lstm.keras"),
    ("Bidirectional LSTM", "dl_bidirectional_lstm.keras"),
    ("Bidirectional GRU", "dl_bidirectional_gru.keras"),
    ("Simple RNN", "dl_simple_rnn.keras"),
]

# ============================================================
# LOAD BANNER
# ============================================================
banner = Image.open("banner.png")

# ============================================================
# DISPLAY BANNER
# ============================================================

st.image(banner, use_container_width=True)

@st.cache_resource(show_spinner="Loading trained deep learning model...")
def load_artifacts():
    tokenizer_path = os.path.join(MODELS_DIR, "dl_tokenizer.pkl")
    label_encoder_path = os.path.join(MODELS_DIR, "dl_label_encoder.pkl")

    if not (os.path.exists(tokenizer_path) and os.path.exists(label_encoder_path)):
        return None  # signals "not trained yet" to the UI below

    chosen_name, chosen_file = None, None
    for name, filename in MODEL_PREFERENCE:
        candidate = os.path.join(MODELS_DIR, filename)
        if os.path.exists(candidate):
            chosen_name, chosen_file = name, candidate
            break

    if chosen_file is None:
        return None

    model = tf.keras.models.load_model(chosen_file)
    tokenizer = joblib.load(tokenizer_path)
    label_encoder = joblib.load(label_encoder_path)

    comparison_path = os.path.join(MODELS_DIR, "dl_comparison_table.csv")
    comparison_df = pd.read_csv(comparison_path) if os.path.exists(comparison_path) else None

    return {
        "model": model,
        "model_name": chosen_name,
        "tokenizer": tokenizer,
        "label_encoder": label_encoder,
        "comparison_df": comparison_df,
    }


def predict_emotion(text, artifacts):
    cleaned = clean_text(text, method="lemmatize")
    X = texts_to_padded([cleaned], artifacts["tokenizer"], maxlen=MAXLEN)

    proba = artifacts["model"].predict(X, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    predicted_label = artifacts["label_encoder"].inverse_transform([pred_idx])[0]

    probs = {}
    for idx, p in enumerate(proba):
        label = artifacts["label_encoder"].inverse_transform([idx])[0]
        probs[label] = float(p)

    return predicted_label, cleaned, probs


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🧠 Emotion Detection — Deep Learning")
st.write(
    "Type a sentence below and a trained recurrent neural network (GRU/LSTM) "
    "will predict the underlying emotion: **joy, sadness, anger, fear, love,** "
    "or **surprise**."
)

artifacts = load_artifacts()

if artifacts is None:
    st.error(
        "No trained deep learning model was found in the `models/` folder.\n\n"
        "This app only loads an already-trained model - it doesn't train one "
        "itself. Please run the training script first:\n\n"
        "```\npython train_dl_models.py\n```\n\n"
        "(and optionally `python tune_dl_model.py` afterwards for the best "
        "result). That creates the `.keras` model files, tokenizer, and "
        "label encoder this app needs. Then re-run `streamlit run app_dl.py`."
    )
    st.stop()

with st.sidebar:
    st.header("ℹ️ About this model")
    st.write(f"**Currently loaded:** {artifacts['model_name']}")
    if artifacts["comparison_df"] is not None:
        st.write("**Model comparison (on held-out test set):**")
        cols = [c for c in ["Model", "Accuracy", "F1-score (weighted)", "Fit Diagnosis"]
                if c in artifacts["comparison_df"].columns]
        st.dataframe(artifacts["comparison_df"][cols].round(4), hide_index=True)
    st.caption(
        "Trained on the dair-ai Emotion dataset (6 classes) using a "
        "recurrent neural network (Simple RNN / LSTM / GRU family), "
        "with the best architecture optionally hyperparameter-tuned via "
        "KerasTuner. See train_dl_models.py, tune_dl_model.py, and the "
        "notebook for the full pipeline."
    )

user_text = st.text_area(
    "Enter a sentence:",
    placeholder="e.g. I can't believe I got the job, this is the best day ever!",
    height=120,
)

col1, col2 = st.columns([1, 4])
with col1:
    predict_clicked = st.button("Predict Emotion", type="primary")

if predict_clicked:
    if not user_text.strip():
        st.warning("Please type a sentence first.")
    else:
        predicted_label, cleaned_text, probs = predict_emotion(user_text, artifacts)
        emoji = EMOTION_EMOJI.get(predicted_label, "")
        color = EMOTION_COLOR.get(predicted_label, "#888888")

        st.markdown(
            f"""
            <div style="padding:20px;border-radius:12px;background-color:{color}22;
                        border:2px solid {color};text-align:center;">
                <h2 style="margin:0;">{emoji} {predicted_label.upper()}</h2>
                <p style="margin:4px 0 0 0;">Confidence: {probs[predicted_label]*100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.subheader("Confidence across all emotions")
        prob_df = (
            pd.DataFrame({"Emotion": list(probs.keys()), "Confidence": list(probs.values())})
            .sort_values("Confidence", ascending=False)
            .set_index("Emotion")
        )
        st.bar_chart(prob_df)

        with st.expander("See how your text was cleaned before prediction"):
            st.write("**Original:**", user_text)
            st.write("**Cleaned (lowercased, stopwords removed, lemmatized):**", cleaned_text or "*(empty after cleaning)*")

st.divider()
st.caption("Built with TensorFlow/Keras + Streamlit | Dataset: dair-ai Emotion dataset")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style='text-align: center;'>
### 🎭 Emotion Detector - Deep Learning"

Built with ❤️ using Streamlit | Developed by nmshah9

</div>
""", unsafe_allow_html=True)
