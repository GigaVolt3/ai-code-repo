"""
HF-005 — AutoModel
==================

Just like AutoTokenizer finds the right tokenizer for a model, AutoModel
finds the right *neural network architecture* for a model id. This lesson
loads a real model and inspects what is inside: the config, the layer
count, the hidden size, and what comes out of a forward pass.

Usage:

    python 05_AutoModel.py                      # defaults (distilbert, CPU)
    python 05_AutoModel.py --model bert-base-uncased   # bigger sibling
    python 05_AutoModel.py --no-head                  # plain AutoModel (no task head)
"""

from __future__ import annotations

import argparse

DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_TEXT = "Transformers are everywhere."


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def demo_config(model) -> None:
    print("=" * 60)
    print("1) model config (the blueprint)")
    print("=" * 60)
    print(f"  architecture  : {model.config.architectures}")
    print(f"  hidden size   : {model.config.hidden_size} (width of every vector)")
    print(f"  layers        : {model.config.num_hidden_layers}")
    print(f"  attention head: {model.config.num_attention_heads}")
    print(f"  vocab size    : {model.config.vocab_size:,}")
    print(f"  parameters    : {count_parameters(model):,} ({count_parameters(model)/1e6:.0f} M)")
    print()


def demo_forward(model, text: str) -> None:
    from transformers import AutoTokenizer

    print("=" * 60)
    print("2) one forward pass")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    encodings = tokenizer(text, return_tensors="pt")
    print(f"  input  : {text!r} -> ids {encodings['input_ids'].tolist()}")
    outputs = model(**encodings)
    hidden = outputs.last_hidden_state
    print(f"  output : {tuple(hidden.shape)}")
    print(f"           (batch, tokens, hidden_size) — one vector per token")
    print()


def demo_with_head(model_id: str, text: str) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("=" * 60)
    print("3) the task head: AutoModelForSequenceClassification")
    print("=" * 60)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    outputs = model(**tokenizer(text, return_tensors="pt"))
    print(f"  logits shape : {tuple(outputs.logits.shape)}  (batch, num_labels)")
    print(f"  num_labels   : {model.config.num_labels}")
    print("  The head maps the [CLS] vector to one score per class.")
    print()


def inspect_only(model_id: str) -> None:
    """Plain AutoModel: no task head, raw hidden states only."""
    from transformers import AutoModel

    print("=" * 60)
    print("plain AutoModel (no task head)")
    print("=" * 60)
    model = AutoModel.from_pretrained(model_id)
    print(f"  loaded {model_id}  ({count_parameters(model):,} parameters)")
    print(f"  last_hidden_state shape: {tuple(model(**model.dummy_inputs).last_hidden_state.shape)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoModel: load any architecture by id.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model id from the Hub")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text to run through the model")
    parser.add_argument("--no-head", action="store_true",
                        help="load plain AutoModel without a task head")
    args = parser.parse_args()

    print("=" * 60)
    print("HF-005 — AutoModel")
    print("=" * 60)
    print("AutoModel reads the config and builds the matching architecture.\n")

    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model)
    print(f"Loaded {args.model} — this is the first real forward pass.\n")

    demo_config(model)
    demo_forward(model, args.text)

    if args.no_head:
        inspect_only(args.model)
    else:
        demo_with_head(args.model, args.text)

    print("=" * 60)
    print("Config -> weights -> forward pass. That is a pretrained model.")
    print("Next lesson HF-006: run inference by hand, step by step.")


if __name__ == "__main__":
    main()