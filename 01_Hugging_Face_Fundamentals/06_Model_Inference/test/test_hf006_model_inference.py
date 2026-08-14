"""HF-006 — Tests for the Model Inference lesson.

Run:  python -m pytest test_hf006_model_inference.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LESSON = Path(__file__).resolve().parents[1] / "06_Model_Inference.py"

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def load_lesson():
    spec = importlib.util.spec_from_file_location("hf006", LESSON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        pytest.skip("torch not installed")


def test_script_exists() -> None:
    assert LESSON.exists()
    text = LESSON.read_text(encoding="utf-8")
    assert "softmax" in text
    assert "HF-006" in text


def test_readme_exists() -> None:
    assert (LESSON.parent / "README.md").exists()


def test_notebook_exists_and_valid() -> None:
    nbformat = pytest.importorskip("nbformat")
    nb = LESSON.parent / "documentation" / "06_Model_Inference.ipynb"
    assert nb.exists(), f"notebook missing: {nb}"
    notebook = nbformat.read(nb, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.get("id") for cell in notebook.cells)


def test_module_imports_and_defaults() -> None:
    mod = load_lesson()
    assert mod.DEFAULT_MODEL == MODEL_ID
    assert mod.labels_of is not None


def test_manual_pipeline_agree() -> None:
    torch = _torch()
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError:
        pytest.skip("transformers not installed")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    text = "Hugging Face is awesome!"
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    manual = int(probs.argmax().item())

    result = pipeline("text-classification", model=MODEL_ID)(text)[0]
    pipe_label = 0 if result["label"] == "NEGATIVE" else 1
    assert manual == pipe_label
    assert abs(float(probs[manual]) - result["score"]) < 1e-3


def test_probs_sum_to_one() -> None:
    torch = _torch()
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    enc = tokenizer("it was okay", return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    assert abs(float(probs.sum()) - 1.0) < 1e-5