# Resources — HF-201 Text Generation

Curated links for going deeper on this lesson.

## Official documentation

- Text generation pipeline — <https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline>
- Generation strategies — <https://huggingface.co/docs/transformers/generation_strategies>
- `generate()` reference — <https://huggingface.co/docs/transformers/main_classes/text_generation>
- GPT-2 model card — <https://huggingface.co/gpt2>

## Courses & tutorials

- Language modeling chapter (HF NLP Course) — <https://huggingface.co/learn/nlp-course/chapter7>
- How to generate text (HF blog) — <https://huggingface.co/blog/how-to-generate>

## Hub models to try

- `gpt2` — 124M causal LM — <https://huggingface.co/gpt2>
- `distilgpt2` — 82M, fast on CPU — <https://huggingface.co/distilgpt2>
- `meta-llama/Llama-3.2-1B` — modern small LLM — <https://huggingface.co/meta-llama/Llama-3.2-1B>
- `bigscience/bloom-560m` — multilingual — <https://huggingface.co/bigscience/bloom-560m>

## Quick reference (this lesson)

| Parameter | Effect |
|-----------|--------|
| `max_new_tokens` | length of the generated continuation |
| `do_sample=True` | enable random sampling instead of greedy |
| `temperature` | randomness (1.0 = default, lower = more focused) |
| `top_k` | keep only the k most probable tokens |
| `top_p` | keep tokens that cover probability mass p |
| `num_return_sequences` | how many continuations to return |
