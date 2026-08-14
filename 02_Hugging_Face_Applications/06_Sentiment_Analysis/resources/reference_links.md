# Resources — HF-206 Sentiment Analysis

Curated links for going deeper on this lesson.

## Official documentation

- Sentiment analysis pipeline — <https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextClassificationPipeline>
- Pipeline overview — <https://huggingface.co/docs/transformers/main_classes/pipelines>
- SST-2 dataset — <https://huggingface.co/datasets/stanfordnlp/sst2>

## Hub models to try

- `distilbert/distilbert-base-uncased-finetuned-sst-2-english` — canonical 2-class model — <https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english>
- `cardiffnlp/twitter-roberta-base-sentiment-latest` — 3-class (pos/neu/neg) — <https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest>
- `j-hartmann/emotion-english-distilroberta-base` — 7 emotion classes — <https://huggingface.co/j-hartmann/emotion-english-distilroberta-base>
- `finiteautomata/bertweet-base-sentiment-analysis` — 3-class on tweets — <https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis>

## Courses & tutorials

- Sentiment analysis demo (HF blog) — <https://huggingface.co/blog/sentiment-analysis-python>
- Text classification chapter (HF NLP Course) — <https://huggingface.co/learn/nlp-course/chapter2>

## Quick reference (this lesson)

| Concept | Meaning |
|---------|---------|
| `sentiment-analysis` | default pipeline task, needs no model argument |
| label | predicted class (`POSITIVE` / `NEGATIVE`, or + `NEUTRAL`) |
| score | model confidence between 0 and 1 |
| batch list | pass a list of strings for bulk analysis |
