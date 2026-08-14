"""
HF-003 — Tokenizer
==================

Models do not understand words — they understand numbers. The tokenizer is
the bridge: it splits text into pieces (tokens), maps each piece to an id,
and can turn ids back into text.

This lesson opens the black box, step by step, on a small and classic
tokenizer (BERT's WordPiece).

Usage:

    python 03_Tokenizer.py                       # defaults (BERT, sample text)
    python 03_Tokenizer.py --text "unhappiness!" # tokenize your own text
    python 03_Tokenizer.py --show-vocab 15       # peek at vocab entries
    python 03_Tokenizer.py --special             # show special tokens only
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):  # Windows cp1252 console cannot print ▁/emoji
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_MODEL = "bert-base-uncased"
DEFAULT_TEXT = "Hugging Face is awesome!"


def load_tokenizer(model: str):
    from transformers import AutoTokenizer

    print(f"Loading tokenizer '{model}' ...")
    return AutoTokenizer.from_pretrained(model)


def demo_pieces(tokenizer, text: str) -> None:
    """The core: text -> list of tokens -> list of ids -> back to text."""
    print("=" * 60)
    print("1) text -> tokens -> ids -> text")
    print("=" * 60)
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"  text   : {text!r}")
    print(f"  tokens : {tokens}")
    print(f"  ids    : {ids}")
    print(f"  decoded: {tokenizer.decode(ids)!r}")
    print()


def demo_special_tokens(tokenizer) -> None:
    """[CLS] [SEP] [PAD] [UNK] — the model's punctuation."""
    print("=" * 60)
    print("2) special tokens")
    print("=" * 60)
    print(f"  [CLS] class start : id {tokenizer.cls_token_id}  {tokenizer.cls_token!r}")
    print(f"  [SEP] separator   : id {tokenizer.sep_token_id}  {tokenizer.sep_token!r}")
    print(f"  [PAD] padding     : id {tokenizer.pad_token_id}  {tokenizer.pad_token!r}")
    print(f"  [UNK] unknown     : id {tokenizer.unk_token_id}  {tokenizer.unk_token!r}")
    print(f"  vocab size        : {tokenizer.vocab_size:,}")
    print()


def demo_subwords(tokenizer) -> None:
    """Long words get split into rarer known pieces."""
    print("=" * 60)
    print("3) subword splitting (unknown words become known pieces)")
    print("=" * 60)
    for word in ["unhappiness", "tokenizer", "transformers", "chatgpt"]:
        tokens = tokenizer.tokenize(word)
        print(f"  {word:<14} -> {tokens}")
    print()


def demo_unknown(tokenizer) -> None:
    """Characters the vocab has never seen end up as [UNK]."""
    print("=" * 60)
    print("4) the [UNK] token")
    print("=" * 60)
    for word in ["\U0001f600", "\u4f60\u597d"]:  # emoji, Chinese characters
        tokens = tokenizer.tokenize(word)
        print(f"  {word!r:<8} -> {tokens}  (ids: {tokenizer.convert_tokens_to_ids(tokens)})")
    print()


def demo_batch(tokenizer, texts: list[str]) -> None:
    """One call, many texts — and a quick look at padding."""
    print("=" * 60)
    print("5) batch encoding (many texts at once)")
    print("=" * 60)
    encodings = tokenizer(texts, padding=True, truncation=True)
    for text, ids in zip(texts, encodings["input_ids"]):
        print(f"  {text!r:<32} -> {ids}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizer basics with BERT WordPiece.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="tokenizer id from the Hub")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text to tokenize")
    parser.add_argument("--show-vocab", type=int, default=0, metavar="N",
                        help="print the first N vocabulary entries (0 = skip)")
    parser.add_argument("--special", action="store_true",
                        help="print only the special-token demo and exit")
    args = parser.parse_args()

    print("=" * 60)
    print("HF-003 — Tokenizer")
    print("=" * 60)
    print("The tokenizer is the bridge between words and numbers.\n")

    tokenizer = load_tokenizer(args.model)

    if args.special:
        demo_special_tokens(tokenizer)
        return

    demo_pieces(tokenizer, args.text)
    demo_special_tokens(tokenizer)
    demo_subwords(tokenizer)
    demo_unknown(tokenizer)
    demo_batch(tokenizer, ["short text", "a much longer text that needs padding"])

    if args.show_vocab:
        print("=" * 60)
        print("6) first entries of the vocabulary")
        print("=" * 60)
        for i, token in enumerate(tokenizer.convert_ids_to_tokens(range(args.show_vocab))):
            print(f"  {i:>4}  {token!r}")
        print()

    print("=" * 60)
    print("Nothing mysterious: text -> tokens -> ids, and back again.")
    print("Next lesson HF-004: AutoTokenizer picks the right tokenizer for you.")


if __name__ == "__main__":
    main()