"""
HF-202 — Summarization
======================

Second lesson of the module: condense long text into a short summary with a
sequence-to-sequence model (BART).

NOTE (transformers v5): the `pipeline("summarization")` helper was removed in
transformers 5.x. The recommended equivalent is the classic three-step
workflow used here — tokenize, generate, decode — which works on every
version and shows what the pipeline used to hide.

What this lesson covers:

    1. AutoTokenizer  -> turn text into input_ids
    2. AutoModelForSeq2SeqLM -> the BART encoder-decoder model
    3. model.generate() -> beam-search decoding for the summary
    4. min_length / max_length / num_beams -> control the summary
    5. long documents -> truncation, because transformers have an input limit

Usage:

    python 02_Summarization.py                                # sample article
    python 02_Summarization.py --file notes.txt               # summarize a file
    python 02_Summarization.py --text "your article text here"
    python 02_Summarization.py --max 60 --min 20              # summary length
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"
MAX_INPUT_TOKENS = 1024  # BART context window
SAMPLE_TEXT = (
    "Hugging Face is a company based in New York that builds open-source tools "
    "for machine learning. Its Transformers library lets developers use thousands "
    "of pretrained models with a few lines of code. The library supports text, "
    "vision and audio tasks, and it runs on PyTorch, TensorFlow and JAX. Models "
    "are shared on the Hugging Face Hub, where the community has uploaded more "
    "than a million artifacts. Because the tools are open source, they are used "
    "by researchers and companies all over the world, from small startups to "
    "large enterprises."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text summarization with a seq2seq model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="seq2seq model id from the Hub")
    parser.add_argument("--text", default=None, help="text to summarize (overrides --file)")
    parser.add_argument("--file", default=None, type=Path, help="read the text from this file")
    parser.add_argument("--min", type=int, default=25, dest="min_length",
                        help="minimum summary length in tokens")
    parser.add_argument("--max", type=int, default=70, dest="max_length",
                        help="maximum summary length in tokens")
    parser.add_argument("--beams", type=int, default=2,
                        help="beam search width (higher = better, slower)")
    return parser.parse_args()


def load_text(args: argparse.Namespace) -> str:
    if args.file:
        return args.file.read_text(encoding="utf-8")
    if args.text:
        return args.text
    return SAMPLE_TEXT


def main() -> None:
    args = parse_args()
    text = load_text(args)

    print("=" * 60)
    print("HF-202 — Summarization")
    print("=" * 60)
    print(f"model        : {args.model}")
    print(f"input length : {len(text.split())} words")
    print(f"summary range: {args.min_length}..{args.max_length} tokens")
    print(f"beams        : {args.beams}")
    print()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    print("Original text:")
    print(f"  {text}")
    print()

    # 1. tokenize  -> 2. generate  -> 3. decode  (the classic inference loop)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=MAX_INPUT_TOKENS)
    summary_ids = model.generate(
        inputs["input_ids"],
        min_length=args.min_length,
        max_length=args.max_length,
        num_beams=args.beams,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("Summary:")
    print(f"  {summary}")
    print()

    print("=" * 60)
    print("Tip: if your document is longer than the model's max input,")
    print("split it into chunks and summarize each chunk, then combine.")
    print("BART handles ~1024 tokens per chunk.")
    print()
    print("transformers v5 note: the summarization *pipeline* was removed;")
    print("tokenize -> generate -> decode is the supported way to do this.")


if __name__ == "__main__":
    main()