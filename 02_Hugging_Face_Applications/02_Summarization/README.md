# HF-202 — Summarization

Second lesson of **Hugging Face Applications**: condense long text into a
short summary with a BART sequence-to-sequence model.

> **transformers v5 note:** the `pipeline("summarization")` helper was removed
> in transformers 5.x. This lesson uses the recommended replacement:
> tokenize → generate → decode with `AutoModelForSeq2SeqLM`. The script works
> on both v4 and v5.

## Folder layout

```
02_Summarization/
|-- 02_Summarization.py   # lesson script (runnable)
|-- documentation/        # lesson docs (teaching notebook)
|   |-- 02_Summarization.ipynb
|   `-- README.md
|-- test/                 # automated checks
|   |-- test_summarization.py
|   `-- README.md
`-- resources/            # links + minimal requirements
    |-- reference_links.md
    `-- requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. summarize the built-in sample article
python 02_Summarization.py

# 2. summarize your own text
python 02_Summarization.py --text "Your long article goes here..."

# 3. or a whole file
python 02_Summarization.py --file notes.txt

# 4. control the summary length and beam width
python 02_Summarization.py --min 20 --max 60 --beams 4

# 5. run the automated checks
pip install pytest
python -m pytest test/ -v
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| `AutoModelForSeq2SeqLM` | BART encoder-decoder model that produces the summary |
| `AutoTokenizer` | turn text into `input_ids` |
| `model.generate()` | beam-search decoding (the summary itself) |
| `min_length` / `max_length` / `num_beams` | summary size and quality |
| `truncation` | long documents — models have an input limit (~1024 tokens for BART) |

The inference loop is only three steps — tokenize, generate, decode —
which is exactly the recipe the removed `summarization` pipeline used
internally.

First run downloads the model (~1.2 GB for `sshleifer/distilbart-cnn-12-6`,
cached once). Use `--model facebook/bart-large-cnn` for better quality on
longer documents, or `google/pegasus-xsum` for news-style headlines.

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-202 | Hugging Face Applications | `02_Summarization.py` | :white_square_button: | :white_square_button: | :white_square_button: |

**Next:** HF-203 — Question Answering.