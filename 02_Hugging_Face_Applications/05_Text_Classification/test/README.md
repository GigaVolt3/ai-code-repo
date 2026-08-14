# Tests — HF-205 Text Classification

Automated checks for the text classification lesson.

## Run

```bash
pip install pytest
python -m pytest test_text_classification.py -v
```

## What is covered

| Test | Verifies |
|------|----------|
| `test_lesson_script_exists` | the lesson script ships and contains the key concepts |
| `test_script_compiles` | the lesson script parses as valid Python |
| `test_documentation_notebook_exists` + `_is_valid` | the lesson notebook exists and validates as nbformat 4 |
| `test_transformers_pipeline_importable` | `pipeline()` is callable |
| `test_text_classification_pipeline_available` | the `text-classification` pipeline is registered |

Tests that need `transformers` **skip** when the package is missing, so the
suite is safe to run before installation.

## Manual check (downloads the model)

```bash
python ../05_Text_Classification.py --text "This product is fantastic!"
```
