"""
HF-203 — Question Answering
===========================

Third lesson of the module: extract the answer to a question from a given
passage of text (extractive QA) with a DistilBERT model fine-tuned on SQuAD.

NOTE (transformers v5): the `pipeline("question-answering")` helper was
removed in transformers 5.x. Here we use the model API directly: the model
predicts a `start` and an `end` token inside the context, and the tokens in
between are the answer.

What this lesson covers:

    1. AutoModelForQuestionAnswering — predicts start/end logits in the text
    2. the (question, context) pair — encoded together as one input
    3. argmax over start_logits / end_logits — locate the answer span
    4. answer reconstruction from input_ids

Usage:

    python 03_Question_Answering.py                    # default Q&A on a sample
    python 03_Question_Answering.py --question "Who won?" --context "Team A won the final."
    python 03_Question_Answering.py --model deepset/roberta-base-squad2
"""

from __future__ import annotations

import argparse

import torch

DEFAULT_MODEL = "distilbert/distilbert-base-cased-distilled-squad"
DEFAULT_CONTEXT = (
    "Hugging Face was founded in 2016 in New York City. The company started as "
    "a chatbot app for teenagers, then pivoted to open-source machine learning "
    "tooling. Its Transformers library, released in 2018, quickly became one of "
    "the most popular libraries in the AI ecosystem, with millions of downloads "
    "per month. The company also hosts the Hugging Face Hub, a platform where "
    "researchers share models, datasets and demos."
)
DEFAULT_QUESTIONS = [
    "When was Hugging Face founded?",
    "Where is the company based?",
    "What did Hugging Face start as?",
    "What does the Hub host?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extractive question answering with a QA model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="QA model id from the Hub")
    parser.add_argument("--question", default=None, help="single question to ask")
    parser.add_argument("--context", default=DEFAULT_CONTEXT, help="passage to search in")
    return parser.parse_args()


def answer_question(model, tokenizer, question: str, context: str) -> str:
    """Ask one question against a context; return the extracted answer span."""
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits, dim=-1).item()
    end_idx = torch.argmax(outputs.end_logits, dim=-1).item()
    if end_idx < start_idx:
        end_idx = start_idx  # guard: end must not precede start

    answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
    return tokenizer.decode(answer_ids, skip_special_tokens=True)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("HF-203 — Question Answering")
    print("=" * 60)
    print(f"model      : {args.model}")
    print()

    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    questions = [args.question] if args.question else DEFAULT_QUESTIONS

    print("Context:")
    print(f"  {args.context}")
    print()

    for question in questions:
        answer = answer_question(model, tokenizer, question, args.context)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print()

    print("=" * 60)
    print("Extractive QA: the answer is a verbatim span of the context.")
    print("The model scores every token as a possible start/end point;")
    print("the span in between is the answer.")
    print()
    print("transformers v5 note: the question-answering *pipeline* was")
    print("removed; this start/end span search is the supported way.")


if __name__ == "__main__":
    main()