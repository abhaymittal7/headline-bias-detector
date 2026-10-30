"""
Measure how accurately bias is detected, using expert-labelled data.

The 5,000 collected RSS headlines have no human labels, so accuracy cannot be
measured on them. Instead we use BABE (datasets/babe_sg2.csv): 3,673 news
sentences that 5 experts labelled "Biased" or "Non-biased".

Three methods are compared on the SAME held-out test set (20%, never used for
training):
  1. Rule-based score   - the project's existing pipeline (spaCy + VADER +
                          lexicon + framing -> compute_bias). A headline counts
                          as "biased" if its label is not "neutral".
  2. TF-IDF + LogReg    - classic machine-learning baseline on word n-grams.
  3. MiniLM + LogReg    - sentence embeddings from all-MiniLM-L6-v2 (the model
                          already used for clustering) + logistic regression.

For methods 2 and 3, 5-fold cross-validation on the full dataset is also
reported, to show the result is stable and not a lucky split.

The trained MiniLM classifier is saved to models/bias_classifier.joblib and is
used by scripts/analyze_headline.py as an extra opinion.

Usage (from the project root):
    python -m scripts.evaluate
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

from src.utils.db import BASE_DIR

DATA_PATH = os.path.join(BASE_DIR, "datasets", "babe_sg2.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "bias_classifier.joblib")
RESULTS_PATH = os.path.join(BASE_DIR, "reports", "evaluation_results.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SEED = 42


# ----------------------------------------
# Data
# ----------------------------------------
def load_babe():
    df = pd.read_csv(DATA_PATH)
    df = df[df["label_bias"].isin(["Biased", "Non-biased"])].reset_index(drop=True)
    texts = df["text"].astype(str).tolist()
    y = (df["label_bias"] == "Biased").astype(int).to_numpy()
    return texts, y


# ----------------------------------------
# Method 1: existing rule-based pipeline
# ----------------------------------------
def rule_based_predict(texts):
    from src.preprocessing.nlp_pipeline import process_text
    from src.preprocessing.entity_mapper import map_entities
    from src.preprocessing.sentiment import get_headline_sentiment, get_entity_sentiment
    from src.features.feature_extractor import extract_features
    from src.models.bias_detector import compute_bias

    preds = []
    for i, text in enumerate(texts, 1):
        nlp = process_text(text)
        entities = map_entities(nlp["entities"])
        sentiment = get_headline_sentiment(text)
        entity_sent = get_entity_sentiment(text, entities)
        features = extract_features(text, sentiment, entity_sent,
                                    nlp["tokens"], nlp["dependencies"], entities)
        preds.append(0 if compute_bias(features)["bias_label"] == "neutral" else 1)
        if i % 200 == 0:
            print(f"    {i}/{len(texts)} sentences scored")
    return np.array(preds)


# ----------------------------------------
# Method 3 helper: sentence embeddings
# ----------------------------------------
def embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(texts, batch_size=64, show_progress_bar=True)


# ----------------------------------------
# Metrics and printing
# ----------------------------------------
def metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def print_confusion(name, cm):
    (tn, fp), (fn, tp) = cm
    print(f"\n  {name}")
    print(f"                        Predicted Non-biased   Predicted Biased")
    print(f"  Actual Non-biased     {tn:>20}   {fp:>16}")
    print(f"  Actual Biased         {fn:>20}   {tp:>16}")


def main():
    t0 = time.time()
    texts, y = load_babe()
    idx = np.arange(len(texts))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    tr_texts = [texts[i] for i in train_idx]
    te_texts = [texts[i] for i in test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    print("=" * 70)
    print(" BIAS DETECTION EVALUATION  (dataset: BABE, expert labels)")
    print("=" * 70)
    print(f"Sentences: {len(texts)}  (Biased {int(y.sum())}, Non-biased {int(len(y) - y.sum())})")
    print(f"Train: {len(train_idx)}   Test (held out): {len(test_idx)}   random_state={SEED}")
    print(f"Always guessing the majority class would score {max(y_te.mean(), 1 - y_te.mean()):.1%} on the test set.")

    results = {}

    print("\n[1/3] Rule-based score (existing pipeline) on the test set...")
    results["Rule-based score"] = {"test": metrics(y_te, rule_based_predict(te_texts))}

    print("\n[2/3] TF-IDF + Logistic Regression...")
    tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(max_iter=2000, C=4.0))
    tfidf.fit(tr_texts, y_tr)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_tfidf = cross_val_score(tfidf, texts, y, cv=cv, scoring="accuracy")
    results["TF-IDF + LogReg"] = {"test": metrics(y_te, tfidf.predict(te_texts)),
                                  "cv_accuracy_mean": round(float(cv_tfidf.mean()), 4),
                                  "cv_accuracy_std": round(float(cv_tfidf.std()), 4)}

    print("\n[3/3] MiniLM embeddings + Logistic Regression...")
    X = embed(texts)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X[train_idx], y_tr)
    cv_emb = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), X, y, cv=cv, scoring="accuracy")
    results["MiniLM + LogReg"] = {"test": metrics(y_te, clf.predict(X[test_idx])),
                                  "cv_accuracy_mean": round(float(cv_emb.mean()), 4),
                                  "cv_accuracy_std": round(float(cv_emb.std()), 4)}

    # ----------------------------------------
    # Report
    # ----------------------------------------
    print("\n" + "=" * 70)
    print(" RESULTS ON HELD-OUT TEST SET")
    print("=" * 70)
    print(f"{'Method':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'5-fold CV acc':>16}")
    print("-" * 70)
    for name, r in results.items():
        m = r["test"]
        cvs = f"{r['cv_accuracy_mean']:.1%} +/- {r['cv_accuracy_std']:.1%}" if "cv_accuracy_mean" in r else "n/a (no training)"
        print(f"{name:<20} {m['accuracy']:>9.1%} {m['precision']:>10.1%} {m['recall']:>8.1%} {m['f1']:>7.1%} {cvs:>16}")

    print("\nConfusion matrices (test set):")
    for name, r in results.items():
        print_confusion(name, r["test"]["confusion_matrix"])

    print("\nNotes:")
    print(" - 'Biased' is the positive class for precision, recall and F1.")
    print(" - Rule-based: any label other than 'neutral' counts as 'Biased'.")
    print(" - BABE contains article sentences, not headlines (related but different domain).")

    # ----------------------------------------
    # Save
    # ----------------------------------------
    import joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"classifier": clf, "embedding_model": EMBEDDING_MODEL,
                 "trained_on": "BABE SG2 train split", "test_accuracy": results["MiniLM + LogReg"]["test"]["accuracy"]},
                MODEL_PATH)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"dataset": "BABE SG2", "n": len(texts), "test_size": len(test_idx),
                   "random_state": SEED, "results": results}, f, indent=2)

    print(f"\nSaved trained model  : {os.path.relpath(MODEL_PATH, BASE_DIR)}")
    print(f"Saved results        : {os.path.relpath(RESULTS_PATH, BASE_DIR)}")
    print(f"Total time: {time.time() - t0:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
