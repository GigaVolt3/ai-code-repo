# Resources — HF-203 Question Answering

Curated links for going deeper on this lesson.

## Official documentation

- Question answering guide — <https://huggingface.co/docs/transformers/tasks/question_answering>
- transformers v5 migration guide (pipeline removals) — <https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md>
- SQuAD dataset — <https://rajpurkar.github.io/SQuAD-explorer/>
- DistilBERT SQuAD model card — <https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad>
- Roberta SQuAD model card — <https://huggingface.co/deepset/roberta-base-squad2>

## Courses & tutorials

- Question answering chapter (HF NLP Course) — <https://huggingface.co/learn/nlp-course/chapter7>
- Extractive QA guide — <https://huggingface.co/docs/transformers/tasks/question_answering>

## Hub models to try

- `distilbert/distilbert-base-cased-distilled-squad` — fast on CPU — <https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad>
- `deepset/roberta-base-squad2` — stronger accuracy — <https://huggingface.co/deepset/roberta-base-squad2>
- `google-bert/bert-large-uncased-whole-word-masking-finetuned-squad` — large BERT — <https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad>

## Quick reference (this lesson)

| Parameter | Effect |
|-----------|--------|
| `question` | the question to answer |
| `context` | the passage the answer must come from |
| `top_k` | number of candidate spans to return |
| `score` | confidence of each candidate span |
| `start` / `end` | character offsets of the span inside the context |
