# Tests — HF-203 Question Answering

Automated checks for the question answering lesson.

## Run

```bash
pip install pytest
python -m pytest test_question_answering.py -v
```

## What is covered

| Test | Verifies |
|------|----------|
| `test_lesson_script_exists` | the lesson script ships and contains the key concepts |
| `test_script_compiles` | the lesson script parses as valid Python |
| `test_documentation_notebook_exists` + `_is_valid` | the lesson notebook exists and validates as nbformat 4 |
| `test_transformers_qa_classes_importable` | `AutoModelForQuestionAnswering` / `AutoTokenizer` importable |
| `test_torch_installed_and_functional` | torch tensors work (needed for inference) |

Tests that need `transformers` / `torch` **skip** when the package is missing,
so the suite is safe to run before installation.

## Manual check (downloads the model)

```bash
python ../03_Question_Answering.py --question "What year was the Eiffel Tower built?" --context "The Eiffel Tower was built in 1889 in Paris."
```