"""
HF-207 — Zero-Shot Classification
=================================

Seventh lesson of the module: classify text into classes the model has NEVER
seen during training. You supply the candidate labels at inference time and a
natural-language inference (NLI) model scores each one.

What this lesson covers:

    1. pipeline("zero-shot-classification", model=...) -> ready-made classifier
    2. candidate labels — define the classes on the fly
    3. multi-label mode — allow several labels at once (multi_label=True)
    4. the trick — NLI: does the text "entail" the label?

Usage:

    python 07_Zero_Shot_Classification.py                      # default example
    python 07_Zero_Shot_Classification.py --text "The stock market rallied today"
    python 07_Zero_Shot_Classification.py --labels "finance,sports,health,technology"
    python 07_Zero_Shot_Classification.py --multi-label
"""

from __future__ import annotations

import argparse

DEFAULT_MODEL = "typeform/distilbert-base-uncased-mnli"
DEFAULT_TEXT = (
    "The central bank raised interest rates again today, citing stubborn "
    "inflation, and markets reacted with a sharp sell-off."
)
DEFAULT_LABELS = ["finance", "sports", "health", "technology"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot classification with Hugging Face pipelines.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="NLI model id from the Hub")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text to classify")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS),
                        help="comma-separated candidate labels")
    parser.add_argument("--multi-label", action="store_true",
                        help="allow several labels at once (sum of scores <= 1)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [label.strip() for label in args.labels.split(",")]

    print("=" * 60)
    print("HF-207 — Zero-Shot Classification")
    print("=" * 60)
    print(f"model   : {args.model}")
    print(f"labels  : {labels}")
    print(f"mode    : {'multi-label' if args.multi_label else 'single-label'}")
    print()

    from transformers import pipeline

    classifier = pipeline("zero-shot-classification", model=args.model, device=-1)

    print(f"Text: {args.text}")
    print()
    result = classifier(
        args.text,
        candidate_labels=labels,
        multi_label=args.multi_label,
    )

    for label, score in zip(result["labels"], result["scores"]):
        print(f"  {label:<12} score={score:.3f}")

    print()
    print("=" * 60)
    print("Zero-shot = labels come from YOU, not from training data.")
    print("The model checks which label the text 'entails' (NLI).")
    if args.multi_label:
        print("Multi-label mode: several labels can be true at once.")


if __name__ == "__main__":
    main()
