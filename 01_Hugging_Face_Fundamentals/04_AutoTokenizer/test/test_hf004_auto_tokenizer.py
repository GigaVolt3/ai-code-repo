"""HF-004 — Tests for the AutoTokenizer lesson.

Run:  python -m pytest test_hf004_auto_tokenizer.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LESSON = Path(__file__).resolve().parents[1] / "04_AutoTokenizer.py"


def load_lesson():
    spec = importlib.util.spec_from_file_location("hf004", LESSON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _auto_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")


def test_script_exists() -> None:
    assert LESSON.exists()
    text = LESSON.read_text(encoding="utf-8")
    assert "AutoTokenizer" in text
    assert "HF-004" in text


def test_readme_exists() -> None:
    assert (LESSON.parent / "README.md").exists()


def test_notebook_exists_and_valid() -> None:
    nbformat = pytest.importorskip("nbformat")
    nb = LESSON.parent / "documentation" / "04_AutoTokenizer.ipynb"
    assert nb.exists(), f"notebook missing: {nb}"
    notebook = nbformat.read(nb, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.get("id") for cell in notebook.cells)


def test_module_imports_and_defaults() -> None:
    mod = load_lesson()
    assert mod.DEFAULT_MODEL == "bert-base-uncased"


def test_families_are_distinct() -> None:
    AutoTokenizer = _auto_tokenizer()
    bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    t5 = AutoTokenizer.from_pretrained("t5-small")
    # tokenization rules really differ between families
    assert bert.tokenize("tokenizer!") != gpt2.tokenize("tokenizer!")
    assert bert.cls_token is not None and gpt2.cls_token is None


def test_padding_and_mask() -> None:
    AutoTokenizer = _auto_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = tokenizer(
        ["short", "a much longer sentence that needs padding"],
        padding="max_length", truncation=True, max_length=12, return_tensors="pt",
    )
    assert enc["input_ids"].shape[1] == 12
    assert enc["attention_mask"].shape == enc["input_ids"].shape
    assert enc["attention_mask"][0].tolist().count(0) > 0  # pad positions masked


def test_save_reload_round_trip(tmp_path) -> None:
    AutoTokenizer = _auto_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(tmp_path)
    reloaded = AutoTokenizer.from_pretrained(tmp_path)
    ids = reloaded("works from a folder too")["input_ids"]
    assert "works from a folder too" in reloaded.decode(ids)