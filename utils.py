"""
utils.py
--------
Shared text-cleaning utilities for the Emotion Detection project.

WHY THIS FILE EXISTS:
The exact same cleaning steps must be applied to text at THREE points in this
project:
  1. When we clean the training data (train_model.py / the notebook)
  2. When we clean the test data for evaluation
  3. When we clean a NEW sentence typed into the Streamlit app at prediction time

If these three used slightly different code, the model would be trained on one
"style" of text and asked to predict on another style -> silently wrong
predictions. Centralising the logic here in one function guarantees
consistency and is a standard MLOps best practice ("training/serving skew"
prevention).
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def ensure_nltk_data():
    """
    Downloads the small NLTK corpora we need (stopword list + lemmatizer
    dictionary) the FIRST time the project is run. On every run after that,
    NLTK finds the files already cached on disk (~/nltk_data) and skips the
    download instantly, so this is safe to call every time the module loads.
    """
    packages = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_data()

STOPWORDS = set(stopwords.words("english"))
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str, method: str = "lemmatize") -> str:
    """
    Cleans one raw sentence and returns a cleaned string ready for
    vectorisation (BoW / TF-IDF) or tokenisation (Word2Vec).

    Steps (in order, and WHY each one matters):
      1. Lowercase              -> "Happy" and "happy" must be treated as the
                                    same word, otherwise the vocabulary size
                                    doubles for no benefit.
      2. Remove non-letters     -> numbers, punctuation, emojis, extra
                                    whitespace add noise and inflate the
                                    vocabulary without adding predictive signal.
      3. Tokenise (split)       -> turns the sentence into a list of words so
                                    each one can be filtered/transformed.
      4. Remove stopwords       -> words like "the", "is", "a" appear in almost
                                    every sentence regardless of emotion, so
                                    they add no discriminative power and only
                                    add dimensionality/noise.
      5. Remove very short tokens (len <= 1) -> leftover single characters
                                    from step 2's regex cleanup.
      6. Stem OR Lemmatize      -> collapse inflected forms of a word
                                    ("crying", "cried", "cries" -> "cri"/"cry")
                                    to the same root so the model doesn't have
                                    to learn each variant separately.
                                      - Stemming is a fast, crude
                                        rule-based chop (Porter Stemmer).
                                      - Lemmatization uses a real dictionary
                                        (WordNet) so the output is always a
                                        real word, at a small speed cost.
                                    We default to lemmatization because it
                                    keeps tokens human-readable and slightly
                                    improves downstream model quality for
                                    short, informal social-media-style text
                                    like this dataset.

    Parameters
    ----------
    text : str
        Raw input sentence.
    method : str
        "lemmatize" (default) or "stem". Lets you switch strategies without
        touching any other code.

    Returns
    -------
    str
        Cleaned, space-joined string, e.g.
        "I am NOT feeling happy today!!" -> "feeling happy today"
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()                          # Step 1
    text = re.sub(r"[^a-z\s]", " ", text)         # Step 2
    tokens = text.split()                         # Step 3
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]  # Steps 4-5

    if method == "stem":                          # Step 6
        tokens = [_stemmer.stem(t) for t in tokens]
    else:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def clean_tokens(text: str, method: str = "lemmatize"):
    """
    Same cleaning as clean_text(), but returns a LIST of tokens instead of a
    joined string. Word2Vec needs tokenised sentences (list-of-words), while
    CountVectorizer/TfidfVectorizer need a single string, hence two helpers.
    """
    return clean_text(text, method=method).split()


def load_emotion_file(path: str):
    """
    Reads one of the dataset files (train.txt / val.txt / test.txt).
    Each line in the raw file looks like:
        "i didnt feel humiliated;sadness"
    i.e. <sentence> ; <emotion label>
    We split on the LAST semicolon (rsplit) in case a sentence itself
    contains a semicolon, so the label is never accidentally cut into the text.
    Returns a pandas DataFrame with columns ["text", "emotion"].
    """
    import pandas as pd

    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text, label = line.rsplit(";", 1)
            texts.append(text)
            labels.append(label)
    return pd.DataFrame({"text": texts, "emotion": labels})
