# Tests — HF-207 Zero-Shot Classification

Automated checks for the zero-shot classification lesson.

## Run

```bash
pip install pytest
python -m pytest test_zero_shot_classification.py -v
```

## What is covered

| Test | Verifies |
|------|----------|
| `test_lesson_script_exists` | the lesson script ships and contains the key concepts |
| `test_script_compiles` | the lesson script parses as valid Python |
| `test_documentation_notebook_exists` + `_is_valid` | the lesson notebook exists and validates as nbformat 4 |
| `test_transformers_pipeline_importable` | `pipeline()` is callable |
| `test_zero_shot_pipeline_available` | the `zero-shot-classification` pipeline is registered |

Tests that need `transformers` **skip** when the package is missing, so the
suite is safe to run before installation.

## Manual check (downloads the model)

```bash
python ../07_Zero_Shot_Classification.py --text "The striker scored in the final minute" --labels "sports,finance,health"
```
