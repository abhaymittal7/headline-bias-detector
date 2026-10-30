"""
Analyze one or more headlines typed by the user, using the SAME pipeline
functions as main.py (spaCy -> entity mapping -> VADER -> features -> bias).
Nothing is written to the database and no internet connection is needed
once the spaCy model and VADER lexicon are installed.

Usage (from the project root):
    python -m scripts.analyze_headline "Opposition slams government over massive fraud"
    python -m scripts.analyze_headline            (interactive mode)
"""
import os
import sys

from src.utils.db import BASE_DIR
from src.preprocessing.nlp_pipeline import process_text
from src.preprocessing.entity_mapper import map_entities
from src.preprocessing.sentiment import get_headline_sentiment, get_entity_sentiment
from src.features.feature_extractor import extract_features
from src.models.bias_detector import compute_bias


MODEL_PATH = os.path.join(BASE_DIR, "models", "bias_classifier.joblib")
_trained = None


def trained_model_opinion(text):
    """Probability of 'Biased' from the classifier trained by scripts.evaluate (if present)."""
    global _trained
    if not os.path.exists(MODEL_PATH):
        return None
    if _trained is None:
        import joblib
        from sentence_transformers import SentenceTransformer
        bundle = joblib.load(MODEL_PATH)
        _trained = (bundle["classifier"], SentenceTransformer(bundle["embedding_model"]), bundle.get("test_accuracy"))
    clf, encoder, acc = _trained
    p = float(clf.predict_proba(encoder.encode([text]))[0][1])
    return p, acc


def analyze(text):
    text = (text or "").strip()
    if not text:
        print("Please enter a non-empty headline.")
        return None

    nlp = process_text(text)
    entities = map_entities(nlp["entities"])
    sentiment = get_headline_sentiment(text)
    entity_sent = get_entity_sentiment(text, entities)
    features = extract_features(
        text, sentiment, entity_sent,
        nlp["tokens"], nlp["dependencies"], entities
    )
    bias = compute_bias(features)

    print("\n" + "=" * 60)
    print(f"HEADLINE : {text}")
    print("-" * 60)
    print("Entities :", ", ".join(f"{e['entity']} ({e['label']}, {e['side']})" for e in entities) or "none")
    print(f"VADER    : compound={sentiment['compound']}  pos={sentiment['pos']}  neg={sentiment['neg']}  neu={sentiment['neu']}")
    print(f"Lexicon  : negative={features['neg_word_count']}  positive={features['pos_word_count']}  intensifiers={features['intensifier_count']}")
    if features["framing"]:
        print("Framing  :", ", ".join(f"{f['entity']}={f['role']}({f['sentiment']})" for f in features["framing"]))
    print(f"RESULT   : {bias['bias_label']}  (score {bias['bias_score']}, thresholds +/-1.5)")
    for reason in bias["explanation"]:
        print(f"   - {reason}")
    opinion = trained_model_opinion(text)
    if opinion:
        p, acc = opinion
        verdict = "Biased" if p >= 0.5 else "Non-biased"
        acc_txt = f", test accuracy {acc:.1%}" if acc is not None else ""
        print(f"TRAINED  : {verdict}  (P(biased) = {p:.2f}; MiniLM + LogReg trained on BABE{acc_txt})")
    print("=" * 60)
    return bias


def main():
    if len(sys.argv) > 1:
        analyze(" ".join(sys.argv[1:]))
        return

    print("Type a headline and press Enter (blank line or Ctrl+C to quit).")
    try:
        while True:
            line = input("\nHeadline> ")
            if not line.strip():
                break
            analyze(line)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
