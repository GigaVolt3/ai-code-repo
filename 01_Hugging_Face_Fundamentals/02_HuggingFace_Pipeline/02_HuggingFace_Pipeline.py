"""
HF-002 — Hugging Face Pipelines
===============================

The pipeline is the fastest way to use any Hugging Face model: one line of
code loads the model, its tokenizer and the post-processing, and one call
does the prediction.

This lesson tours the most useful pipeline tasks with small CPU-friendly
models, so you can feel how the same recipe works everywhere.

Usage:

    python 02_HuggingFace_Pipeline.py                    # tour of all tasks
    python 02_HuggingFace_Pipeline.py --task sentiment   # one task only
    python 02_HuggingFace_Pipeline.py --device 0         # use GPU (if you have one)

Tasks: sentiment | classification | zero-shot | generation | all
"""

from __future__ import annotations

import argparse
import time


def timed(label: str, fn):
    """Run fn() and print how long it took."""
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    print(f"[{label}] finished in {dt:.1f}s\n")
    return result


def demo_sentiment(device: int) -> None:
    from transformers import pipeline

    print("=" * 60)
    print("1) sentiment-analysis — is the text positive or negative?")
    print("=" * 60)
    sent = timed(
        "load", lambda: pipeline("sentiment-analysis", device=device)
    )
    texts = [
        "I love this product, it works perfectly!",
        "The delivery took two weeks and the box was damaged.",
        "It is fine, nothing special.",
    ]
    for text, r in zip(texts, sent(texts)):
        print(f"  {r['label']:<9} ({r['score']:.2f})  {text!r}")
    print()


def demo_classification(device: int) -> None:
    from transformers import pipeline

    print("=" * 60)
    print("2) text-classification — same idea, explicit model choice")
    print("=" * 60)
    classifier = timed(
        "load",
        lambda: pipeline(
            "text-classification",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
        ),
    )
    text = "The acting was brilliant but the ending felt rushed."
    result = classifier(text, top_k=2)
    if result and isinstance(result[0], list):  # v5 batches even single texts
        result = result[0]
    for r in result:
        print(f"  {r['label']:<9} ({r['score']:.2f})")
    print()


def demo_zero_shot(device: int) -> None:
    from transformers import pipeline

    print("=" * 60)
    print("3) zero-shot-classification — you invent the categories")
    print("=" * 60)
    classifier = timed(
        "load",
        lambda: pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=device,
        ),
    )
    text = "The central bank raised interest rates again this morning."
    result = classifier(text, candidate_labels=["finance", "sports", "health"])
    for label, score in zip(result["labels"], result["scores"]):
        print(f"  {label:<10} {score:.3f}")
    print()


def demo_generation(device: int) -> None:
    from transformers import GenerationConfig, pipeline

    print("=" * 60)
    print("4) text-generation — the model continues your sentence")
    print("=" * 60)
    generator = timed(
        "load",
        lambda: pipeline("text-generation", model="distilgpt2", device=device),
    )
    generation_config = GenerationConfig(
        max_new_tokens=25,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
    )
    outputs = generator("Once upon a time", generation_config=generation_config)
    print(f"  prompt   : 'Once upon a time'")
    print(f"  continued: {outputs[0]['generated_text']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tour of Hugging Face pipeline tasks.")
    parser.add_argument(
        "--task",
        default="all",
        choices=["sentiment", "classification", "zero-shot", "generation", "all"],
        help="which pipeline task to demonstrate (default: all)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="device to run on: -1 = CPU (default), 0 = first GPU",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("HF-002 — Hugging Face Pipelines")
    print("=" * 60)
    print("The pipeline is one line of code that hides the whole model stack.")
    print("On first run each model is downloaded once and then cached.\n")

    if args.task in ("all", "sentiment"):
        demo_sentiment(args.device)
    if args.task in ("all", "classification"):
        demo_classification(args.device)
    if args.task in ("all", "zero-shot"):
        demo_zero_shot(args.device)
    if args.task in ("all", "generation"):
        demo_generation(args.device)

    print("=" * 60)
    print("Same recipe everywhere: pick a task, pick a model, call it.")
    print("Next lesson HF-003 looks under the hood: the tokenizer.")


if __name__ == "__main__":
    main()