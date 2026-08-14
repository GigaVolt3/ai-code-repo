"""HF-005 — Tests for the AutoModel lesson.

Run:  python -m pytest test_hf005_auto_model.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LESSON = Path(__file__).resolve().parents[1] / "05_AutoModel.py"


def load_lesson():
    spec = importlib.util.spec_from_file_location("hf005", LESSON)
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
    assert "AutoModel" in text
    assert "HF-005" in text


def test_readme_exists() -> None:
    assert (LESSON.parent / "README.md").exists()


def test_notebook_exists_and_valid() -> None:
    nbformat = pytest.importorskip("nbformat")
    nb = LESSON.parent / "documentation" / "05_AutoModel.ipynb"
    assert nb.exists(), f"notebook missing: {nb}"
    notebook = nbformat.read(nb, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.get("id") for cell in notebook.cells)


def test_module_imports_and_defaults() -> None:
    mod = load_lesson()
    assert mod.DEFAULT_MODEL == "distilbert-base-uncased"
    assert mod.count_parameters is not None


def test_forward_pass_shape() -> None:
    torch = _torch()
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    enc = tokenizer("Transformers are everywhere.", return_tensors="pt")
    out = model(**enc)
    assert tuple(out.last_hidden_state.shape) == (1, 6, 768)
    assert isinstance(out.last_hidden_state, torch.Tensor)


def test_config_blueprint() -> None:
    try:
        from transformers import AutoModel
    except ImportError:
        pytest.skip("transformers not installed")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    c = model.config
    assert c.hidden_size == 768
    assert c.num_hidden_layers == 6
    assert c.num_attention_heads == 12
    assert c.vocab_size == 30522


def test_head_variant_logits() -> None:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    logits = model(**tokenizer("hello", return_tensors="pt")).logits
    assert tuple(logits.shape) == (1, 2)