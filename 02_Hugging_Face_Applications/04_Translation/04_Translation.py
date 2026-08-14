"""
HF-204 — Translation
====================

Fourth lesson of the module: translate text between languages with the
Helsinki-NLP OPUS-MT models.

NOTE (transformers v5): the `pipeline("translation_en_to_fr")` helper was
removed in transformers 5.x. The recommended equivalent is the classic
three-step workflow used here — tokenize, generate, decode — which works on
every version.

What this lesson covers:

    1. AutoTokenizer — encode the source sentence
    2. AutoModelForSeq2SeqLM — the OPUS-MT encoder-decoder model
    3. model.generate() — beam-search decoding for the translation
    4. one model per language pair — choose the right one on the Hub

Usage:

    python 04_Translation.py                                  # en -> fr
    python 04_Translation.py --target de                      # en -> de
    python 04_Translation.py --text "Machine learning is fun."
    python 04_Translation.py --model Helsinki-NLP/opus-mt-en-hi   # any pair
    python 04_Translation.py --file readme.md                 # translate a file
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_TEXT = (
    "Hugging Face is a company that builds open-source tools for artificial "
    "intelligence. Its libraries are used by millions of developers around the "
    "world to train and run machine learning models."
)

LANGS = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Machine translation with OPUS-MT models.")
    parser.add_argument("--model", default=None, help="OPUS-MT model id (overrides --target)")
    parser.add_argument("--target", default="fr", choices=sorted(LANGS),
                        help="target language for English source text")
    parser.add_argument("--text", default=None, help="text to translate")
    parser.add_argument("--file", default=None, type=Path, help="read text from this file")
    return parser.parse_args()


def load_text(args: argparse.Namespace) -> str:
    if args.file:
        return args.file.read_text(encoding="utf-8")
    if args.text:
        return args.text
    return DEFAULT_TEXT


def main() -> None:
    args = parse_args()

    model_id = args.model or LANGS[args.target]
    text = load_text(args)

    print("=" * 60)
    print("HF-204 — Translation")
    print("=" * 60)
    print(f"model   : {model_id}")
    print()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    print(f"Source:")
    print(f"  {text}")
    print()

    # 1. tokenize  -> 2. generate  -> 3. decode
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    translated_ids = model.generate(**inputs, max_new_tokens=512)
    translation = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    print("Translation:")
    print(f"  {translation}")
    print()

    print("=" * 60)
    print("Note: Helsinki-NLP OPUS-MT is one model pair per language.")
    print("Pick the right model id on the Hub for the pair you need.")
    print()
    print("transformers v5 note: the translation *pipeline* was removed;")
    print("tokenize -> generate -> decode is the supported way to do this.")


if __name__ == "__main__":
    main()