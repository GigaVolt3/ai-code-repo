# HF-204 — Translation

Fourth lesson of **Hugging Face Applications**: translate text between
languages with Helsinki-NLP OPUS-MT models.

> **transformers v5 note:** the `pipeline("translation_en_to_fr")` helper was
> removed in transformers 5.x. This lesson uses the recommended replacement:
> tokenize → generate → decode with `AutoModelForSeq2SeqLM`. The script works
> on both v4 and v5.

## Folder layout

```
04_Translation/
|-- 04_Translation.py   # lesson script (runnable)
|-- documentation/      # lesson docs (teaching notebook)
|   |-- 04_Translation.ipynb
|   `-- README.md
|-- test/               # automated checks
|   |-- test_translation.py
|   `-- README.md
`-- resources/          # links + minimal requirements
    |-- reference_links.md
    `-- requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. translate the sample text to French (default)
python 04_Translation.py

# 2. choose another target language (de, es, hi)
python 04_Translation.py --target de

# 3. translate your own text
python 04_Translation.py --text "Machine learning is fun."

# 4. translate a whole file
python 04_Translation.py --file notes.txt

# 5. use any pair directly
python 04_Translation.py --model Helsinki-NLP/opus-mt-en-zh

# 6. run the automated checks
pip install pytest
python -m pytest test/ -v
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| `AutoModelForSeq2SeqLM` | OPUS-MT encoder-decoder model |
| `AutoTokenizer` | encode the source sentence into `input_ids` |
| `model.generate()` | beam-search decoding for the translation |
| one model per pair | OPUS-MT has a separate model for every language pair |

The inference loop is the same three steps as in HF-202 — tokenize,
generate, decode — which is exactly the recipe the removed `translation`
pipeline used internally.

First run downloads the model (~300 MB for OPUS-MT en-fr, cached once).

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-204 | Hugging Face Applications | `04_Translation.py` | :white_square_button: | :white_square_button: | :white_square_button: |

**Next:** HF-205 — Text Classification.