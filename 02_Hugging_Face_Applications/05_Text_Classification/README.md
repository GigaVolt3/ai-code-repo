# HF-205 — Text Classification

Fifth lesson of **Hugging Face Applications**: classify text into fixed
categories with a DistilBERT model fine-tuned on SST-2.

## Folder layout

```
05_Text_Classification/
|-- 05_Text_Classification.py   # lesson script (runnable)
|-- documentation/              # lesson docs (teaching notebook)
|   |-- 05_Text_Classification.ipynb
|   `-- README.md
|-- test/                       # automated checks
|   |-- test_text_classification.py
|   `-- README.md
`-- resources/                  # links + minimal requirements
    |-- reference_links.md
    `-- requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. classify the built-in sample texts
python 05_Text_Classification.py

# 2. classify your own text
python 05_Text_Classification.py --text "I love this movie"

# 3. see the top-2 classes with scores
python 05_Text_Classification.py --text "It was okay, I guess." --top-k 2

# 4. swap in a different classifier
python 05_Text_Classification.py --model cardiffnlp/twitter-roberta-base-sentiment-latest

# 5. run the automated checks
pip install pytest
python -m pytest test/ -v
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| `pipeline("text-classification")` | ready-made classification API |
| single-label vs. multi-label | one class vs. several classes per text |
| batch input | pass a list of texts, get predictions in one call |
| `top_k` | how many class probabilities to show |

First run downloads the model (~270 MB, cached once).

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-205 | Hugging Face Applications | `05_Text_Classification.py` | :white_square_button: | :white_square_button: | :white_square_button: |

**Next:** HF-206 — Sentiment Analysis.
