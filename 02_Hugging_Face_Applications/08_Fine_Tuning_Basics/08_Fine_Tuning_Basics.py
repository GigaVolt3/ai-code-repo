"""
HF-208 — Fine-Tuning Basics
===========================

Eighth lesson of the module: the first step beyond pretrained models. We
fine-tune a small DistilBERT model on a tiny labeled dataset (built into this
script, no extra downloads) using the `Trainer` API, then save, reload and
compare it with the pretrained base.

What this lesson covers:

    1. AutoModelForSequenceClassification — adds a classification head
    2. AutoTokenizer + a small tokenization function
    3. Trainer / TrainingArguments — the standard fine-tuning loop
    4. trainer.save_model() / from_pretrained() — persist and reload
    5. before vs. after — the fine-tuned model is better on our task

Usage:

    python 08_Fine_Tuning_Basics.py                 # full run (CPU-friendly)
    python 08_Fine_Tuning_Basics.py --epochs 3      # train a bit longer
    python 08_Fine_Tuning_Basics.py --save-dir models/fine-tuned
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_SAVE_DIR = Path("models/fine-tuned")

# A tiny handcrafted sentiment dataset: (text, label) with 1 = positive, 0 = negative.
TRAIN_DATA = [
    ("this product is amazing and works perfectly", 1),
    ("i love this app it makes my life easier", 1),
    ("great quality and fast delivery", 1),
    ("excellent customer service", 1),
    ("wonderful experience overall", 1),
    ("the best purchase i ever made", 1),
    ("fantastic and highly recommended", 1),
    ("superb and worth every penny", 1),
    ("terrible product broke on the first day", 0),
    ("i hate this app it keeps crashing", 0),
    ("awful quality and very slow delivery", 0),
    ("poor customer service", 0),
    ("horrible experience overall", 0),
    ("the worst purchase i ever made", 0),
    ("useless and not recommended", 0),
    ("disappointing and overpriced", 0),
]

EVAL_DATA = [
    ("this works great", 1),
    ("i am very happy with it", 1),
    ("love it, works perfectly", 1),
    ("total waste of money", 0),
    ("very bad product", 0),
    ("do not buy this, it is terrible", 0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a small model with the Trainer API.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="base model id from the Hub")
    parser.add_argument("--epochs", type=int, default=6, help="training epochs")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, type=Path,
                        help="where to save the fine-tuned model")
    return parser.parse_args()


class SentimentDataset(torch.utils.data.Dataset):
    """Small torch Dataset returning dict samples (what the Trainer expects)."""

    def __init__(self, encodings, labels):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def build_dataset(data: list[tuple[str, int]]):
    """Return (texts, labels) lists; kept dependency-free (no `datasets` package)."""
    texts = [t for t, _ in data]
    labels = [l for _, l in data]
    return texts, labels


def tokenize(tokenizer, texts: list[str]):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


def evaluate(model, tokenizer, texts: list[str], labels: list[int], device) -> float:
    """Return accuracy on the given labeled texts."""
    inputs = tokenize(tokenizer, texts)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = logits.argmax(dim=-1).cpu().tolist()
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("HF-208 — Fine-Tuning Basics")
    print("=" * 60)
    print(f"base model : {args.model}")
    print(f"train size : {len(TRAIN_DATA)} examples")
    print(f"eval size  : {len(EVAL_DATA)} examples")
    print(f"epochs     : {args.epochs}")
    print(f"save dir   : {args.save_dir}")
    print()

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    # Seed everything so the demo is reproducible run-to-run.
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    device = model.device

    train_texts, train_labels = build_dataset(TRAIN_DATA)
    eval_texts, eval_labels = build_dataset(EVAL_DATA)

    base_acc = evaluate(model, tokenizer, eval_texts, eval_labels, device)
    print(f"Accuracy BEFORE fine-tuning: {base_acc:.2%}")
    print()

    train_dataset = SentimentDataset(tokenize(tokenizer, train_texts), train_labels)

    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Training ...")
    trainer.train()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Saved fine-tuned model to: {args.save_dir}")
    print()

    fine = AutoModelForSequenceClassification.from_pretrained(args.save_dir)
    fine_acc = evaluate(fine, tokenizer, eval_texts, eval_labels, fine.device)
    print(f"Accuracy AFTER fine-tuning: {fine_acc:.2%}")
    print(f"Gain: {fine_acc - base_acc:+.2%}")
    print()

    print("=" * 60)
    print("Fine-tuning adapts a general model to YOUR task with a small")
    print("labeled dataset. The saved model can be reloaded with")
    print("AutoModelForSequenceClassification.from_pretrained() and used")
    print("in any pipeline, exactly like a hub model.")


if __name__ == "__main__":
    main()
