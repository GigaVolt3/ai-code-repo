"""HF-003 — Tests for the Tokenizer lesson.

Run:  python -m pytest test_hf003_tokenizer.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LESSON = Path(__file__).resolve().parents[1] / "03_Tokenizer.py"


def load_lesson():
    spec = importlib.util.spec_from_file_location("hf003", LESSON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")


def test_script_exists() -> None:
    assert LESSON.exists()
    text = LESSON.read_text(encoding="utf-8")
    assert "tokenize" in text
    assert "HF-003" in text


def test_readme_exists() -> None:
    assert (LESSON.parent / "README.md").exists()


def test_notebook_exists_and_valid() -> None:
    nbformat = pytest.importorskip("nbformat")
    nb = LESSON.parent / "documentation" / "03_Tokenizer.ipynb"
    assert nb.exists(), f"notebook missing: {nb}"
    notebook = nbformat.read(nb, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.get("id") for cell in notebook.cells)


def test_module_imports_and_defaults() -> None:
    mod = load_lesson()
    assert mod.DEFAULT_MODEL == "bert-base-uncased"
    assert "awesome" in mod.DEFAULT_TEXT


def test_round_trip_encode_decode() -> None:
    AutoTokenizer = _tokenizer()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ids = tokenizer("Hugging Face is awesome!")["input_ids"]
    decoded = tokenizer.decode(ids)
    assert "hugging face is awesome" in decoded


def test_special_token_ids() -> None:
    AutoTokenizer = _tokenizer()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    assert tokenizer.cls_token_id == 101
    assert tokenizer.sep_token_id == 102
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 100
    assert tokenizer.vocab_size == 30522


def test_subword_splitting() -> None:
    AutoTokenizer = _tokenizer()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize("unhappiness")
    assert len(tokens) >= 2
    assert tokens[0] == "un"
    assert tokens[-1].startswith("##")
