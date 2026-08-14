# Resources — HF-204 Translation

Curated links for going deeper on this lesson.

## Official documentation

- Translation guide — <https://huggingface.co/docs/transformers/tasks/translation>
- transformers v5 migration guide (pipeline removals) — <https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md>
- OPUS-MT overview (Helsinki-NLP) — <https://github.com/Helsinki-NLP/Opus-MT>

## Hub models to try

- `Helsinki-NLP/opus-mt-en-fr` — English to French — <https://huggingface.co/Helsinki-NLP/opus-mt-en-fr>
- `Helsinki-NLP/opus-mt-en-de` — English to German — <https://huggingface.co/Helsinki-NLP/opus-mt-en-de>
- `Helsinki-NLP/opus-mt-en-es` — English to Spanish — <https://huggingface.co/Helsinki-NLP/opus-mt-en-es>
- `Helsinki-NLP/opus-mt-en-hi` — English to Hindi — <https://huggingface.co/Helsinki-NLP/opus-mt-en-hi>
- Browse all OPUS-MT models — <https://huggingface.co/Helsinki-NLP>
- `facebook/nllb-200-distilled-600M` — one model, 200 languages — <https://huggingface.co/facebook/nllb-200-distilled-600M>
- `t5-small` — generic seq2seq, can be prompted for translation — <https://huggingface.co/t5-small>

## Courses & tutorials

- Translation chapter (HF NLP Course) — <https://huggingface.co/learn/nlp-course/chapter9>

## Quick reference (this lesson)

| Parameter | Effect |
|-----------|--------|
| `task` | `translation_en_to_fr` etc. — encodes source and target |
| `model` | the OPUS-MT model id (one per language pair) |
| `truncation=True` | clip input longer than the model context |
| `max_length` | cap the translation length |
