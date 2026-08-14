# Resources — HF-202 Summarization

Curated links for going deeper on this lesson.

## Official documentation

- Summarization guide — <https://huggingface.co/docs/transformers/tasks/summarization>
- Seq2seq training guide — <https://huggingface.co/docs/transformers/training>
- transformers v5 migration guide (pipeline removals) — <https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md>
- BART model card — <https://huggingface.co/facebook/bart-large-cnn>
- distilbart (fast CNN distillate) — <https://huggingface.co/sshleifer/distilbart-cnn-12-6>

## Courses & tutorials

- Summarization tutorial (HF blog) — <https://huggingface.co/blog/summarization>
- Seq2seq chapter (HF NLP Course) — <https://huggingface.co/learn/nlp-course/chapter9>

## Hub models to try

- `sshleifer/distilbart-cnn-12-6` — fast, CPU-friendly — <https://huggingface.co/sshleifer/distilbart-cnn-12-6>
- `facebook/bart-large-cnn` — stronger quality — <https://huggingface.co/facebook/bart-large-cnn>
- `google/pegasus-xsum` — news headline style — <https://huggingface.co/google/pegasus-xsum>
- `Falconsai/text_summarization` — small T5 fine-tune — <https://huggingface.co/Falconsai/text_summarization>

## Quick reference (this lesson)

| Parameter | Effect |
|-----------|--------|
| `min_length` | shortest allowed summary (tokens) |
| `max_length` | longest allowed summary (tokens) |
| `truncation=True` | clip input longer than the model context |
| `do_sample` | sampling vs. beam search (beam is default for summarization) |
