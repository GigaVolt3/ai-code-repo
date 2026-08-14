"""
HF-206 — Sentiment Analysis
===========================

Sixth lesson of the module: determine the emotional tone (positive / negative)
of a piece of text. Sentiment analysis is the classic special case of text
classification, and `pipeline("sentiment-analysis")` gives it a dedicated API.

What this lesson covers:

    1. pipeline("sentiment-analysis") — the default pipeline task
    2. label + score — confidence of each prediction
    3. batch processing — score many texts in one call
    4. default model vs. explicit model — same engine, spelled out
    5. the difference between 2-class and 3-class sentiment models

Usage:

    python 06_Sentiment_Analysis.py                         # default sample texts
    python 06_Sentiment_Analysis.py --text "I am so excited for the weekend"
    python 06_Sentiment_Analysis.py --model cardiffnlp/twitter-roberta-base-sentiment-latest
"""

from __future__ import annotations

import argparse

DEFAULT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_TEXTS = [
    "I absolutely loved the film, the acting was brilliant.",
    "The service was slow and the food was cold.",
    "The product works exactly as described, very happy.",
    "I would not recommend this to anyone.",
    "It was okay, nothing special but nothing terrible either.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentiment analysis with Hugging Face pipelines.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="sentiment model id (default is the canonical SST-2 DistilBERT)")
    parser.add_argument("--text", default=None, help="single text to analyze")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("HF-206 — Sentiment Analysis")
    print("=" * 60)
    print(f"model : {args.model}")
    print()

    from transformers import pipeline

    # No model argument needed: sentiment-analysis is the default pipeline task.
    sentiment = pipeline("sentiment-analysis", model=args.model, device=-1)

    texts = [args.text] if args.text else DEFAULT_TEXTS
    results = sentiment(texts)

    for text, r in zip(texts, results):
        label_up = r["label"].upper()
        # 3-class models use NEGATIVE/NEUTRAL/POSITIVE, 2-class models POSITIVE/NEGATIVE.
        mood = "positive" if "POS" in label_up else "negative" if "NEG" in label_up else "neutral"
        print(f"{mood:<9} ({r['score']:.3f})  {text!r}")

    print()
    print("=" * 60)
    print("Sentiment analysis is text classification for emotions.")
    print("2-class models (SST-2) give POSITIVE/NEGATIVE;")
    print("3-class models (e.g. twitter-roberta) add NEUTRAL.")


if __name__ == "__main__":
    main()
