# Tests — HF-204 Translation

Automated checks for the translation lesson.

## Run

```bash
pip install pytest
python -m pytest test_translation.py -v
```

## What is covered

| Test | Verifies |
|------|----------|
| `test_lesson_script_exists` | the lesson script ships and contains the key concepts |
| `test_script_compiles` | the lesson script parses as valid Python |
| `test_documentation_notebook_exists` + `_is_valid` | the lesson notebook exists and validates as nbformat 4 |
| `test_transformers_seq2seq_classes_importable` | `AutoModelForSeq2SeqLM` / `AutoTokenizer` importable |
| `test_torch_installed_and_functional` | torch tensors work (needed for inference) |

Tests that need `transformers` / `torch` **skip** when the package is missing,
so the suite is safe to run before installation.

## Manual check (downloads the model)

```bash
python ../04_Translation.py --text "The weather is nice today."
```