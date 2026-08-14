# HF-206 — Sentiment Analysis

Sixth lesson of **Hugging Face Applications**: determine the emotional tone
of text with the classic sentiment-analysis pipeline.

## Folder layout

```
06_Sentiment_Analysis/
|-- 06_Sentiment_Analysis.py   # lesson script (runnable)
|-- documentation/             # lesson docs (teaching notebook)
|   |-- 06_Sentiment_Analysis.ipynb
|   `-- README.md
|-- test/                      # automated checks
|   |-- test_sentiment_analysis.py
|   `-- README.md
`-- resources/                 # links + minimal requirements
    |-- reference_links.md
    `-- requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. analyze the built-in sample texts
python 06_Sentiment_Analysis.py

# 2. analyze your own text
python 06_Sentiment_Analysis.py --text "I am so excited for the weekend!"

# 3. try a 3-class model that also knows NEUTRAL
python 06_Sentiment_Analysis.py --model cardiffnlp/twitter-roberta-base-sentiment-latest

# 4. run the automated checks
pip install pytest
python -m pytest test/ -v
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| `pipeline("sentiment-analysis")` | the canonical example task of the whole library |
| label + score | predicted class and confidence |
| batch processing | score many texts in one call |
| 2-class vs. 3-class | POSITIVE/NEGATIVE vs. POSITIVE/NEUTRAL/NEGATIVE |
| default task | `sentiment-analysis` works with no model argument at all |

First run downloads the model (~270 MB, cached once).

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-206 | Hugging Face Applications | `06_Sentiment_Analysis.py` | :white_square_button: | :white_square_button: | :white_square_button: |

**Next:** HF-207 — Zero-Shot Classification.
