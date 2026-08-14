# Tests - HF-006 Model Inference

Automated checks for the Model Inference lesson.

## Run

```bash
pip install pytest
python -m pytest test_hf006_model_inference.py -v
```

(root of the test suite: `python -m pytest test/ -v` from the lesson folder)

## What is covered

| Test | Verifies |
|------|----------|
| `test_script_exists` | the lesson script ships and contains the key concepts |
| `test_readme_exists` | the lesson README is present |
| `test_notebook_exists_and_valid` | the lesson notebook exists and validates as nbformat 4 |
| behavior tests | manual probs sum to 1, manual label == pipeline label, scores match to 1e-3 |

Tests that need `transformers` / `torch` / `huggingface_hub` **skip** when the
package is missing; model-dependent tests use the shared cache, so the first
run may download a small tokenizer or model.

## Manual check (downloads cached models)

```bash
python ../06_Model_Inference.py --text "terrible experience" --print-logits
```
