"""
HF-204 — Tests for the Translation lesson.

Run from this folder:

    python -m pytest test_translation.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

LESSON_DIR = Path(__file__).resolve().parents[1]
SCRIPT = LESSON_DIR / "04_Translation.py"


def test_lesson_script_exists() -> None:
    assert SCRIPT.exists(), f"lesson script missing: {SCRIPT}"
    text = SCRIPT.read_text(encoding="utf-8")
    assert "AutoModelForSeq2SeqLM" in text
    assert "opus-mt" in text
    assert "model.generate" in text


def test_script_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    compile(source, str(SCRIPT), "exec")


def test_documentation_notebook_exists() -> None:
    nb = LESSON_DIR / "documentation" / "04_Translation.ipynb"
    assert nb.exists(), f"lesson notebook missing: {nb}"


def test_documentation_notebook_is_valid() -> None:
    """The lesson notebook must parse and validate as nbformat 4."""
    nbformat = pytest.importorskip("nbformat", reason="nbformat not installed — pip install nbformat")
    nb_path = LESSON_DIR / "documentation" / "04_Translation.ipynb"
    notebook = nbformat.read(nb_path, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.get("id") for cell in notebook.cells), "cells are missing id fields"


def test_transformers_seq2seq_classes_importable() -> None:
    transformers = pytest.importorskip("transformers", reason="transformers not installed")
    assert hasattr(transformers, "AutoModelForSeq2SeqLM")
    assert hasattr(transformers, "AutoTokenizer")


def test_torch_installed_and_functional() -> None:
    torch = pytest.importorskip("torch", reason="torch not installed")
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.sum().item() == 6.0, "torch is installed but tensors do not work"