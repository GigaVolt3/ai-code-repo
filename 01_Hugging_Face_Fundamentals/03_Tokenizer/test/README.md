# Tests - HF-003 Tokenizer

Automated checks for the Tokenizer lesson.

## Run

```bash
pip install pytest
python -m pytest test_hf003_tokenizer.py -v
```

(root of the test suite: `python -m pytest test/ -v` from the lesson folder)

## What is covered

| Test | Verifies |
|------|----------|
| `test_script_exists` | the lesson script ships and contains the key concepts |
| `test_readme_exists` | the lesson README is present |
| `test_notebook_exists_and_valid` | the lesson notebook exists and validates as nbformat 4 |
| behavior tests | BERT tokenizer round trip works, special-token ids correct, subwords split |

Tests that need `transformers` / `torch` / `huggingface_hub` **skip** when the
package is missing; model-dependent tests use the shared cache, so the first
run may download a small tokenizer or model.

## Manual check (downloads cached models)

```bash
python ../03_Tokenizer.py --text "unhappiness!" --show-vocab 8
```
