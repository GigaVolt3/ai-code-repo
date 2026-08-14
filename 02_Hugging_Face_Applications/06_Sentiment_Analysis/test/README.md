# Tests — HF-206 Sentiment Analysis

Automated checks for the sentiment analysis lesson.

## Run

```bash
pip install pytest
python -m pytest test_sentiment_analysis.py -v
```

## What is covered

| Test | Verifies |
|------|----------|
| `test_lesson_script_exists` | the lesson script ships and contains the key concepts |
| `test_script_compiles` | the lesson script parses as valid Python |
| `test_documentation_notebook_exists` + `_is_valid` | the lesson notebook exists and validates as nbformat 4 |
| `test_transformers_pipeline_importable` | `pipeline()` is callable |
| `test_sentiment_analysis_pipeline_available` | the `sentiment-analysis` pipeline is registered |

Tests that need `transformers` **skip** when the package is missing, so the
suite is safe to run before installation.

## Manual check (downloads the model)

```bash
python ../06_Sentiment_Analysis.py --text "The hotel room was spotless and the staff were wonderful."
```
