# Tests — HF-208 Fine-Tuning Basics

Automated checks for the fine-tuning lesson.

## Run

```bash
pip install pytest
python -m pytest test_fine_tuning_basics.py -v
```

## What is covered

| Test | Verifies |
|------|----------|
| `test_lesson_script_exists` | the lesson script ships and contains the key concepts |
| `test_script_compiles` | the lesson script parses as valid Python |
| `test_documentation_notebook_exists` + `_is_valid` | the lesson notebook exists and validates as nbformat 4 |
| `test_dataset_is_builtin` | the script defines its own labeled dataset |
| `test_transformers_trainer_importable` | `Trainer` / `TrainingArguments` are importable |
| `test_torch_installed_and_functional` | torch tensors work (needed for training) |

Tests that need `transformers` / `torch` **skip** when the package is missing,
so the suite is safe to run before installation.

## Manual check (downloads the base model, trains ~1-2 min)

```bash
python ../08_Fine_Tuning_Basics.py --epochs 6
```
