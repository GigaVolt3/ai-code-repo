"""
HF-004 — AutoTokenizer
======================

Every model family tokenizes differently (BERT splits words into subwords
with ##, GPT-2 has its own byte-level BPE, T5 uses SentencePiece...).
AutoTokenizer reads the model's config and loads the *right* tokenizer for
that model automatically.

This lesson compares three tokenizer families and shows how to save a
tokenizer to a folder and reload it.

Usage:

    python 04_AutoTokenizer.py                          # defaults (BERT)
    python 04_AutoTokenizer.py --compare                # BERT vs GPT-2 vs T5
    python 04_AutoTokenizer.py --model gpt2 --text "hello"   # any model id
    python 04_AutoTokenizer.py --save-temp              # save/reload demo
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows cp1252 console cannot print ▁
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_MODEL = "bert-base-uncased"
DEFAULT_TEXT = "Hugging Face tokenizers are fast!"


def load(model: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer


def show_family(model: str, text: str) -> None:
    tokenizer = load(model)
    tokens = tokenizer.tokenize(text)
    ids = tokenizer(text)["input_ids"]
    print(f"  {'-' * 56}")
    print(f"  {model}")
    print(f"  tokens      : {tokens}")
    print(f"  ids         : {ids}")
    print(f"  decode back : {tokenizer.decode(ids)!r}")
    print(f"  vocab size  : {tokenizer.vocab_size:,}")
    print(f"  special cls : {tokenizer.cls_token!r}  sep: {tokenizer.sep_token!r}  pad: {tokenizer.pad_token!r}")
    print()


def demo_one(model: str, text: str) -> None:
    tokenizer = load(model)
    print("=" * 60)
    print("text -> tokens -> ids -> text")
    print("=" * 60)
    ids = tokenizer(text)["input_ids"]
    print(f"  text   : {text!r}")
    print(f"  ids    : {ids}")
    print(f"  tokens : {tokenizer.convert_ids_to_tokens(ids)}")
    print(f"  decode : {tokenizer.decode(ids)!r}")
    print()


def demo_batch(model: str) -> None:
    tokenizer = load(model)
    print("=" * 60)
    print("padding and truncation in one call")
    print("=" * 60)
    texts = [
        "short",
        "a medium length sentence here",
        "a very long sentence that will be truncated by the max_length limit set below",
    ]
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=12,
        return_tensors="pt",
    )
    print(f"  tensor shape : {tuple(encodings['input_ids'].shape)}  (texts x max_length)")
    print(f"  attention    : {encodings['attention_mask'].tolist()}")
    print("  (attention_mask tells the model which positions are real words)")
    print()


def demo_save_reload() -> None:
    """Save a tokenizer to a folder and reload it — a real workflow."""
    print("=" * 60)
    print("save/reload round trip")
    print("=" * 60)
    tokenizer = load(DEFAULT_MODEL)
    with tempfile.TemporaryDirectory(prefix="hf004-") as tmp:
        save_dir = Path(tmp) / "my_tokenizer"
        tokenizer.save_pretrained(save_dir)
        print(f"  saved {tokenizer.vocab_size:,} vocab + config to {save_dir.name}/")
        for f in sorted(p.name for p in save_dir.iterdir()):
            print(f"    - {f}")

        from transformers import AutoTokenizer

        reloaded = AutoTokenizer.from_pretrained(save_dir)
        ids = reloaded("reloaded from a folder just like from the Hub")["input_ids"]
        print(f"  reloaded works: {reloaded.decode(ids)!r}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoTokenizer: the right tokenizer, automatically.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="any model id from the Hub")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text to tokenize")
    parser.add_argument("--compare", action="store_true",
                        help="compare BERT vs GPT-2 vs T5 tokenizer families")
    parser.add_argument("--save-temp", action="store_true",
                        help="demo saving a tokenizer to a temp folder and reloading")
    args = parser.parse_args()

    print("=" * 60)
    print("HF-004 — AutoTokenizer")
    print("=" * 60)
    print("AutoTokenizer reads the model config and loads the matching tokenizer.\n")

    if args.compare:
        for model in ["bert-base-uncased", "gpt2", "t5-small"]:
            show_family(model, "tokenizer!");
        print("  Same text, different tokenizers — that is why AutoTokenizer exists.")
        print()

    demo_batch(args.model)
    if args.save_temp:
        demo_save_reload()

    print("=" * 60)
    print("One class, every model family covered.")
    print("Next lesson HF-005: AutoModel loads the matching model the same way.")


if __name__ == "__main__":
    main()