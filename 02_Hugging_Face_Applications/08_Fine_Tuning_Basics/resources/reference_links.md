# Resources — HF-208 Fine-Tuning Basics

Curated links for going deeper on this lesson.

## Official documentation

- Trainer API — <https://huggingface.co/docs/transformers/main_classes/trainer>
- TrainingArguments — <https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments>
- Auto classes — <https://huggingface.co/docs/transformers/model_doc/auto>
- Fine-tuning with Trainer (guide) — <https://huggingface.co/docs/transformers/training>
- Text classification guide — <https://huggingface.co/docs/transformers/tasks/text_classification>

## Courses & tutorials

- Fine-tuning chapter (HF NLP Course) — <https://huggingface.co/learn/nlp-course/chapter3>
- Fine-tune BERT for sentiment (HF blog) — <https://huggingface.co/blog/bert-101>

## Hub resources

- `distilbert-base-uncased` — the base model used here — <https://huggingface.co/distilbert/distilbert-base-uncased>
- Full fine-tuned reference: `distilbert/distilbert-base-uncased-finetuned-sst-2-english` — <https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english>

## Quick reference (this lesson)

| Piece | Purpose |
|-------|---------|
| `AutoModelForSequenceClassification` | backbone + new classification head |
| `AutoTokenizer` | tokenize texts for the model |
| `Trainer` | training loop with batching, gradients, checkpointing |
| `TrainingArguments` | epochs, batch size, logging, output dir |
| `save_model()` | persist weights + config to a folder |
| `from_pretrained()` | reload a saved folder like a hub model |
