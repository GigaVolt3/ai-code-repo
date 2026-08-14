# HF-207 — Zero-Shot Classification

Seventh lesson of **Hugging Face Applications**: classify text into classes
the model has never seen, by supplying candidate labels at inference time.

## Folder layout

```
07_Zero_Shot_Classification/
|-- 07_Zero_Shot_Classification.py   # lesson script (runnable)
|-- documentation/                   # lesson docs (teaching notebook)
|   |-- 07_Zero_Shot_Classification.ipynb
|   `-- README.md
|-- test/                            # automated checks
|   |-- test_zero_shot_classification.py
|   `-- README.md
`-- resources/                       # links + minimal requirements
    |-- reference_links.md
    `-- requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. classify the default text against the default labels
python 07_Zero_Shot_Classification.py

# 2. classify your own text
python 07_Zero_Shot_Classification.py --text "The striker scored in the final minute"

# 3. define your own labels
python 07_Zero_Shot_Classification.py --labels "politics,sports,weather,crime"

# 4. allow several labels at once
python 07_Zero_Shot_Classification.py --multi-label

# 5. run the automated checks
pip install pytest
python -m pytest test/ -v
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| `pipeline("zero-shot-classification")` | classify without training data |
| `candidate_labels` | classes defined at inference time |
| NLI trick | the model scores "text entails label" |
| `multi_label=True` | several labels may be true simultaneously |
| label + score | sorted list of candidate labels with confidence |

First run downloads the model (~270 MB for the DistilBERT MNLI model,
cached once). Use `--model facebook/bart-large-mnli` for stronger accuracy.

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-207 | Hugging Face Applications | `07_Zero_Shot_Classification.py` | :white_square_button: | :white_square_button: | :white_square_button: |

**Next:** HF-208 — Fine-Tuning Basics.
