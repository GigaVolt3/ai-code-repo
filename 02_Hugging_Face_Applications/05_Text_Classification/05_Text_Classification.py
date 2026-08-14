"""
HF-205 — Text Classification
============================

Fifth lesson of the module: classify text into fixed categories (single-label
classification) with a DistilBERT model fine-tuned on SST-2.

What this lesson covers:

    1. pipeline("text-classification", model=...) -> ready-made classifier
    2. single-label vs. multi-label classification
    3. classifying a whole batch of texts in one call
    4. top_k — how many class probabilities to return

Usage:

    python 05_Text_Classification.py                       # default sample texts
    python 05_Text_Classification.py --text "I love this movie"
    python 05_Text_Classification.py --text "This was a waste of time" --top-k 2
    python 05_Text_Classification.py --model cardiffnlp/twitter-roberta-base-sentiment-latest
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
    parser = argparse.ArgumentParser(description="Text classification with Hugging Face pipelines.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="classifier model id from the Hub")
    parser.add_argument("--text", default=None, help="single text to classify")
    parser.add_argument("--top-k", type=int, default=1, dest="top_k",
                        help="how many classes to report per text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("HF-205 — Text Classification")
    print("=" * 60)
    print(f"model : {args.model}")
    print()

    from transformers import pipeline

    classifier = pipeline("text-classification", model=args.model, device=-1)

    texts = [args.text] if args.text else DEFAULT_TEXTS
    results = classifier(texts, top_k=args.top_k)

    # Normalize: older/newer transformers return dicts, lists of dicts, or
    # nested lists depending on (batch, top_k). Flatten to a list of dicts.
    for text, preds in zip(texts, results):
        if isinstance(preds, dict):
            preds = [preds]
        elif preds and isinstance(preds[0], list):
            preds = preds[0]
        print(f"Text: {text!r}")
        for p in preds:
            print(f"  -> {p['label']:<12} score={p['score']:.3f}")
        print()

    print("=" * 60)
    print("Single-label model: one class per text. For multi-label,")
    print("use a model trained for it and pass top_k=None to see all.")


if __name__ == "__main__":
    main()
