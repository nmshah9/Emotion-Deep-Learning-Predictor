"""
build_final_comparison.py
==========================
STEP vii (assignment) — Final comparison table across EVERY model built in
both phases of this project: classic ML (Logistic Regression / Decision
Tree / Random Forest, from the first project), Deep Learning (Simple RNN /
LSTM / GRU / their Bidirectional & Stacked variants / KerasTuner-tuned GRU),
and Transformers (BERT / DistilBERT / RoBERTa, if that script was run
successfully on a machine with internet access).

Run with:   python build_final_comparison.py

WHY a single combined table matters: each phase used a different family of
techniques (bag-of-words counts vs. sequential neural nets vs. pretrained
transformers), but they were all evaluated the same way — accuracy,
precision, recall, and F1-score (weighted) on the SAME held-out test.txt
file that none of them were trained or tuned on. That shared yardstick is
what makes a fair, apples-to-apples "which approach actually works best on
this dataset" conclusion possible.
"""

import os
import pandas as pd

DL_DIR = os.path.dirname(os.path.abspath(__file__))
ML_MODELS_DIR = os.path.join(os.path.dirname(DL_DIR), "emo_proj", "models")
# ^ Adjust this path if you've placed the classic-ML project folder
#   somewhere else relative to this one — it just needs to point at the
#   "models" folder produced by the first project's train_model.py.

DL_MODELS_DIR = os.path.join(DL_DIR, "models")

COLUMNS = ["Phase", "Model", "Accuracy", "Precision (weighted)",
           "Recall (weighted)", "F1-score (weighted)", "Fit Diagnosis", "Notes"]


def load_ml_results():
    path = os.path.join(ML_MODELS_DIR, "comparison_table.csv")
    if not os.path.exists(path):
        print(f"(Classic-ML comparison table not found at {path} - skipping. "
              f"Run the first project's train_model.py, or update ML_MODELS_DIR "
              f"in this script if that project lives somewhere else.)")
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    df["Phase"] = "Classic ML"
    df["Fit Diagnosis"] = "N/A (not applicable to non-iterative models)"
    df["Notes"] = "Bag-of-Words features, GridSearch/RandomizedSearch-tuned"
    return df[COLUMNS]


def load_dl_results():
    path = os.path.join(DL_MODELS_DIR, "dl_comparison_table.csv")
    if not os.path.exists(path):
        print(f"(Deep-learning comparison table not found at {path} - "
              f"run train_dl_models.py and tune_dl_model.py first.)")
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    df["Phase"] = "Deep Learning"
    # NOTE: "Best Hyperparameters" only exists once tune_dl_model.py has run
    # (it's the column that records the tuned model's winning config).
    # If only train_dl_models.py has been run so far, this column is
    # missing entirely from the CSV - df.get() then returns the literal
    # default value "" (a plain string) instead of a Series, and calling
    # .fillna() on a string crashes with the AttributeError seen in the
    # Streamlit traceback. We guard for that explicitly here.
    if "Best Hyperparameters" in df.columns:
        notes = df["Best Hyperparameters"].fillna("Default architecture, no tuning")
        notes = notes.replace("", "Default architecture, no tuning")
    else:
        notes = pd.Series(["Default architecture, no tuning"] * len(df), index=df.index)
    df["Notes"] = notes
    return df[COLUMNS]


def load_transformer_results():
    path = os.path.join(DL_MODELS_DIR, "transformer_comparison_table.csv")
    if not os.path.exists(path):
        print(f"(Transformer comparison table not found at {path}. "
              f"This is expected if train_transformers.py could not reach "
              f"huggingface.co in this environment - see README_DL.md. "
              f"Run it on a machine with internet access to populate this "
              f"section for real.)")
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    df["Phase"] = "Transformer (pretrained)"
    df["Fit Diagnosis"] = "N/A (see training logs)"
    df["Notes"] = "Fine-tuned from Hugging Face pretrained checkpoint"
    return df[COLUMNS]


if __name__ == "__main__":
    ml_df = load_ml_results()
    dl_df = load_dl_results()
    tf_df = load_transformer_results()

    combined = pd.concat([ml_df, dl_df, tf_df], ignore_index=True)

    if combined.empty:
        print("No results found from any phase - nothing to combine. "
              "Run the training scripts first.")
    else:
        combined = combined.sort_values("F1-score (weighted)", ascending=False).reset_index(drop=True)
        combined.insert(0, "Rank", range(1, len(combined) + 1))

        print("\n" + "=" * 100)
        print("FINAL MASTER COMPARISON — ALL MODELS, ALL PHASES")
        print("=" * 100)
        print(combined.to_string(index=False))

        out_path = os.path.join(DL_MODELS_DIR, "FINAL_master_comparison_table.csv")
        combined.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

        best = combined.iloc[0]
        print(f"\n>>> BEST MODEL OVERALL: {best['Model']} ({best['Phase']})")
        print(f"    F1-weighted = {best['F1-score (weighted)']:.4f}, "
              f"Accuracy = {best['Accuracy']:.4f}")

        if tf_df.empty:
            print("\nNOTE: Transformer results are not included above because "
                  "train_transformers.py could not download pretrained weights "
                  "in this environment. Given that BERT/DistilBERT/RoBERTa "
                  "start from massive pretrained language knowledge rather "
                  "than learning English from scratch on ~18,000 sentences, "
                  "they would be expected to match or modestly exceed the "
                  "best Deep Learning result above once fine-tuned on a "
                  "machine with internet access - re-run this script after "
                  "train_transformers.py completes successfully to confirm "
                  "with real numbers rather than an assumption.")
