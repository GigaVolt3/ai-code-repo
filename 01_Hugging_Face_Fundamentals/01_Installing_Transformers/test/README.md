# Tests — HF-001 Installing Transformers

Automated checks that verify the environment is ready for the course.

## Run

```bash
pip install pytest
python -m pytest test_installing_transformers.py -v
```

## What is covered

| Test | Verifies |
|------|----------|
| `test_python_version_supported` | Python 3.8+ |
| `test_smoke_test_script_exists` | the lesson script exists and ships a smoke test |
| `test_documentation_notebook_exists` | the lesson notebook exists |
| `test_documentation_notebook_is_valid` | the notebook parses and validates as nbformat 4 |
| `test_transformers_installed` | `transformers` is importable |
| `test_transformers_version_supported` | `transformers` >= 4.40 |
| `test_transformers_pipeline_importable` | `pipeline()` is callable |
| `test_torch_installed_and_functional` | `torch` imports and tensors work |
| `test_cuda_check_returns_bool` | `torch.cuda.is_available()` works |

Tests that need `transformers` / `torch` / `nbformat` **skip** when the package
is missing, so the suite is safe to run before installation (expect 4 passed
+ 5 skipped without the deps; 9 passed with them installed).

## Manual checks (not automated)

- GPU detection: `python -c "import torch; print(torch.cuda.is_available())"`
- Smoke test: `python ../01_Installing_Transformers.py --smoke`
