# Resources — HF-207 Zero-Shot Classification

Curated links for going deeper on this lesson.

## Official documentation

- Zero-shot classification pipeline — <https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.ZeroShotClassificationPipeline>
- Zero-shot classification guide — <https://huggingface.co/docs/transformers/tasks/zero_shot_classification>
- MNLI dataset — <https://huggingface.co/datasets/nyu-mll/multi_nli>

## Hub models to try

- `typeform/distilbert-base-uncased-mnli` — fast on CPU — <https://huggingface.co/typeform/distilbert-base-uncased-mnli>
- `facebook/bart-large-mnli` — stronger accuracy, bigger — <https://huggingface.co/facebook/bart-large-mnli>
- `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` — SOTA zero-shot — <https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0>
- `cross-encoder/nli-deberta-v3-base` — cross-encoder NLI — <https://huggingface.co/cross-encoder/nli-deberta-v3-base>

## Courses & tutorials

- Zero-shot classification explainer (HF blog) — <https://huggingface.co/blog/zero-shot>
- NLI chapter (HF NLP Course) — <https://huggingface.co/learn/nlp-course/chapter7>

## Quick reference (this lesson)

| Parameter | Effect |
|-----------|--------|
| `candidate_labels` | the classes to score (you define them) |
| `multi_label` | allow several labels true at once |
| `entailment` | the NLI mechanism behind zero-shot scoring |
| label + score | labels sorted by confidence, highest first |
