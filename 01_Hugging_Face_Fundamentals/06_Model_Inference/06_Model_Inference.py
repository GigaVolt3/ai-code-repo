"""
HF-006 — Model Inference
========================

The pipeline (HF-002) hides a fixed recipe. This lesson does inference by
hand — tokenize, forward pass, logits, softmax, argmax — so you can see
exactly what happens between your text and the predicted label.

We use the sentiment model seen in HF-002/05, so the download is cached.

Usage:

    python 06_Model_Inference.py                      # defaults
    python 06_Model_Inference.py --text "terrible"    # your own text
    python 06_Model_Inference.py --print-logits       # show all raw scores
"""

from __future__ import annotations

import argparse

import torch

DEFAULT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_TEXT = "Hugging Face is awesome!"


def load_pieces(model_id: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def labels_of(model) -> dict[int, str]:
    if model.config.id2label:
        return model.config.id2label
    return {0: "NEGATIVE", 1: "POSITIVE"}


def step_by_step(tokenizer, model, text: str) -> None:
    print("=" * 60)
    print("1) the recipe, one step at a time")
    print("=" * 60)

    # Step 1 — tokenize: words -> ids (HF-003)
    encodings = tokenizer(text, return_tensors="pt")
    print(f"  step 1  tokenize   : ids {encodings['input_ids'].tolist()}")
    print(f"          tokens     : {tokenizer.convert_ids_to_tokens(encodings['input_ids'][0])}")

    # Step 2 — forward pass: ids -> logits (raw scores, not probabilities)
    with torch.no_grad():
        logits = model(**encodings).logits
    print(f"  step 2  forward    : logits {logits.tolist()}")

    # Step 3 — softmax: logits -> probabilities
    probs = torch.softmax(logits, dim=-1)
    print(f"  step 3  softmax    : probs {probs.tolist()}  (they sum to 1)")

    # Step 4 — argmax: probabilities -> predicted class id
    pred_id = int(probs.argmax(dim=-1).item())
    label = labels_of(model).get(pred_id, f"id{pred_id}")
    print(f"  step 4  argmax     : class {pred_id} -> '{label}'")
    print()


def demo_manual_label(tokenizer, model, text: str) -> None:
    labels = labels_of(model)
    probs = manual_probs(tokenizer, model, text)
    best = int(probs.argmax().item())
    print("=" * 60)
    print("2) probability table")
    print("=" * 60)
    for cls_id, p in enumerate(probs.tolist()):
        print(f"  {labels.get(cls_id, f'class {cls_id}'):<10} {p:.4f}")
    print(f"  -> {labels.get(best, best)} with {probs[best]:.3f}")
    print()


def manual_probs(tokenizer, model, text: str) -> torch.Tensor:
    encodings = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encodings).logits
    return torch.softmax(logits, dim=-1)[0]


def demo_compare(tokenizer, model, text: str) -> None:
    from transformers import pipeline

    print("=" * 60)
    print("3) same model, same result via pipeline")
    print("=" * 60)
    pipe = pipeline("text-classification", model=model.name_or_path)
    result = pipe(text, top_k=None)[0]
    print(f"  pipeline: {result}")
    manual = manual_probs(tokenizer, model, text)
    print(f"  manual  : labels={labels_of(model)}, probs={[round(p, 4) for p in manual.tolist()]}")
    print("  (the pipeline just wraps steps 1-4 — the numbers match)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Model inference by hand: the 4 steps.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model id from the Hub")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text to classify")
    parser.add_argument("--print-logits", action="store_true",
                        help="also print the raw (pre-softmax) scores")
    args = parser.parse_args()

    print("=" * 60)
    print("HF-006 — Model Inference")
    print("=" * 60)
    print("Tokenize -> forward -> softmax -> argmax. Four steps, every time.\n")

    tokenizer, model = load_pieces(args.model)

    step_by_step(tokenizer, model, args.text)
    demo_manual_label(tokenizer, model, args.text)
    if args.print_logits:
        encodings = tokenizer(args.text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**encodings).logits
        print(f"  raw logits: {logits.tolist()}  (any sign, not yet probabilities)\n")
    demo_compare(tokenizer, model, args.text)

    print("=" * 60)
    print("Now you know what the pipeline does under the hood.")
    print("Next lesson HF-007: load pretrained models in more ways.")


if __name__ == "__main__":
    main()