"""
HF-201 — Text Generation
========================

Second module of the course. First real application: generate text with a
pretrained causal language model (GPT-2) using the `pipeline` API.

What this lesson covers:

    1. pipeline("text-generation", model=...) -> a ready-made generator
    2. sampling parameters: max_new_tokens, temperature, top_k, top_p
    3. deterministic vs. creative decoding (greedy vs. sampling)
    4. generating several candidate continuations for one prompt

Usage:

    python 01_Text_Generation.py                     # defaults (gpt2, sample)
    python 01_Text_Generation.py --prompt "Once upon a time" --max 30 --n 3
    python 01_Text_Generation.py --model distilgpt2   # smaller/faster model
    python 01_Text_Generation.py --prompt "2+2 is" --greedy   # deterministic
    python 01_Text_Generation.py --temperature 0.2   # more focused output
"""

from __future__ import annotations

import argparse

DEFAULT_MODEL = "gpt2"
DEFAULT_PROMPT = "Artificial intelligence will change the world because"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text generation with Hugging Face pipelines.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="causal LM id from the Hub")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="start of the text to continue")
    parser.add_argument("--max", type=int, default=40, dest="max_new_tokens",
                        help="max new tokens to generate")
    parser.add_argument("--n", type=int, default=3, help="number of continuations to sample")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="sampling temperature (lower = more focused)")
    parser.add_argument("--greedy", action="store_true",
                        help="use greedy decoding (deterministic, no sampling)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("HF-201 — Text Generation")
    print("=" * 60)
    print(f"model      : {args.model}")
    print(f"prompt     : {args.prompt!r}")
    print(f"decoding   : {'greedy' if args.greedy else 'sampling'}")
    if not args.greedy:
        print(f"temperature: {args.temperature}")
    print(f"continuations: {args.n}")
    print()

    from transformers import pipeline

    generator = pipeline(
        "text-generation",
        model=args.model,
        truncation=True,
        device=-1,
    )

    if args.greedy:
        print("Greedy decoding — the model always picks the single most probable token.\n")
        from transformers import GenerationConfig

        generation_config = GenerationConfig(max_new_tokens=args.max_new_tokens)
        outputs = generator(args.prompt, generation_config=generation_config)
        for i, out in enumerate(outputs, 1):
            print(f"--- continuation {i} ---")
            print(out["generated_text"])
            print()
        return

    print("Sampling decoding — each run draws a different continuation "
          "(temperature controls how random).\n")
    from transformers import GenerationConfig

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.n,
        do_sample=True,
        temperature=args.temperature,
        top_k=50,
        top_p=0.95,
    )
    outputs = generator(args.prompt, generation_config=generation_config)
    for i, out in enumerate(outputs, 1):
        print(f"--- continuation {i} ---")
        print(out["generated_text"])
        print()

    print("=" * 60)
    print("Lower temperature -> more predictable text.")
    print("Higher temperature -> more creative (and more nonsense) text.")


if __name__ == "__main__":
    main()
