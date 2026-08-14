"""
HF-007 — Loading Pretrained Models
==================================

Hugging Face models live on the Hub, but you can also:
  * download them once and share the folder (offline-friendly),
  * pin a specific revision (commit hash, branch, tag),
  * save your own fine-tuned models to disk,
  * load them from a local folder exactly like from the Hub.

This lesson exercises all of those.

Usage:

    python 07_Loading_Pretrained_Model.py                    # defaults
    python 07_Loading_Pretrained_Model.py --model bert-base-uncased
    python 07_Loading_Pretrained_Model.py --revision main
    python 07_Loading_Pretrained_Model.py --offline          # Hub disabled
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

DEFAULT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def demo_from_id(model_id: str, revision: str) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("=" * 60)
    print("1) load directly from the Hub")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
    print(f"  model : {model_id}  (revision '{revision}')")
    print(f"  config: {model.config.model_type} with {model.config.num_labels} labels")
    print()


def demo_local_folder(model_id: str) -> None:
    """snapshot_download -> load from the local folder (no Hub involved)."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("=" * 60)
    print("2) download once -> load from a local folder")
    print("=" * 60)
    with tempfile.TemporaryDirectory(prefix="hf007-") as tmp:
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=tmp,
            ignore_patterns=["*.bin", "*.h5", "*.ot", "onnx/*", "map.jpeg"],
        )
        print(f"  snapshot in : {local_dir}")
        print(f"  files       : {sorted(Path(local_dir).iterdir())}")

        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForSequenceClassification.from_pretrained(local_dir)
        result = model(**tokenizer("works from a folder too!", return_tensors="pt"))
        print(f"  forward pass: logits shape {tuple(result.logits.shape)} — works offline")
    print()


def demo_save_roundtrip(model_id: str) -> None:
    """save_pretrained -> reload: the exact workflow for fine-tuned models."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("=" * 60)
    print("3) save and reload (what HF-208 fine-tuning ends with)")
    print("=" * 60)
    with tempfile.TemporaryDirectory(prefix="hf007-") as tmp:
        save_dir = Path(tmp) / "my_model"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"  saved to {save_dir.name}/: {sorted(p.name for p in save_dir.iterdir())}")

        reloaded = AutoModelForSequenceClassification.from_pretrained(save_dir)
        print(f"  reloaded from disk, same API, same weights")
        print(f"  parameters: {sum(p.numel() for p in reloaded.parameters()):,}")
    print()


def demo_revisions(model_id: str, revision: str) -> None:
    print("=" * 60)
    print("4) revisions: pin your model version")
    print("=" * 60)
    print(f"  from_pretrained('{model_id}', revision='{revision}')")
    print("  — revision can be a branch, a tag, or a commit hash.")
    print("  — models on the Hub are snapshots: the weights never change behind you.")
    print()


def demo_offline(model_id: str) -> None:
    print("=" * 60)
    print("5) offline mode")
    print("=" * 60)
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        print(f"  HF_HUB_OFFLINE=1 -> loaded '{model_id}' from cache, no network.")
        print(f"  (works because lesson 1 downloaded it already)")
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ways to load pretrained models.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model id from the Hub")
    parser.add_argument("--revision", default="main", help="branch/tag/commit to pin")
    parser.add_argument("--offline", action="store_true",
                        help="only the offline-from-cache demo")
    args = parser.parse_args()

    print("=" * 60)
    print("HF-007 — Loading Pretrained Models")
    print("=" * 60)
    print("One API — from_pretrained — for the Hub, a folder, or a revision.\n")

    if args.offline:
        demo_offline(args.model)
        return

    demo_from_id(args.model, args.revision)
    demo_local_folder(args.model)
    demo_save_roundtrip(args.model)
    demo_revisions(args.model, args.revision)
    demo_offline(args.model)

    print("=" * 60)
    print("from_pretrained covers Hub, folder, revision — one function.")
    print("Next lesson HF-008: where models live on disk (the cache).")


if __name__ == "__main__":
    main()