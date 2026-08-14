# Resources — HF-205 Text Classification

Curated links for going deeper on this lesson.

## Official documentation

- Text classification pipeline — <https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextClassificationPipeline>
- Text classification guide — <https://huggingface.co/docs/transformers/tasks/text_classification>
- SST-2 dataset — <https://huggingface.co/datasets/stanfordnlp/sst2>
- DistilBERT SST-2 model card — <https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english>

## Hub models to try

- `distilbert/distilbert-base-uncased-finetuned-sst-2-english` — fast on CPU — <https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english>
- `cardiffnlp/twitter-roberta-base-sentiment-latest` — 3-class sentiment (pos/neu/neg) — <https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest>
- `roberta-large-openai-detector` — AI-generated text detector — <https://huggingface.co/roberta-large-openai-detector>
- `j-hartmann/emotion-english-distilroberta-base` — 7 emotions — <https://huggingface.co/j-hartmann/emotion-english-distilroberta-base>

## Courses & tutorials

- Text classification chapter (HF NLP Course) — <https://huggingface.co/learn/nlp-course/chapter2>

## Quick reference (this lesson)

| Parameter | Effect |
|-----------|--------|
| `top_k` | how many classes to return per text |
| `batch` | pass a list of texts to classify them together |
| label + score | class name and probability of the prediction |
