"""
HF-001 — Tests for the Transformers installation lesson.
=========================================================

Run from this folder:

    python -m pytest test_installing_transformers.py -v

Tests that need `transformers` / `torch` skip automatically when the package
is missing, so the suite is safe to run before and after installation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MIN_PYTHON = (3, 8)
MIN_TRANSFORMERS = (4, 40)


def _transformers():
    try:
        import transformers
        return transformers
    except ImportError:
        pytest.skip("transformers not installed — run: pip install transformers")


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        pytest.skip("torch not installed — run: pip install torch")


# --- environment checks (always run) ---------------------------------------

def test_python_version_supported() -> None:
    assert sys.version_info >= MIN_PYTHON, "Python 3.8+ is required"


def test_smoke_test_script_exists() -> None:
    lesson = Path(__file__).resolve().parents[1] / "01_Installing_Transformers.py"
    assert lesson.exists(), f"lesson script missing: {lesson}"
    text = lesson.read_text(encoding="utf-8")
    assert "pip install transformers" in text
    assert "smoke" in text  # the script ships with an optional smoke test


def test_documentation_notebook_exists() -> None:
    nb = Path(__file__).resolve().parents[1] / "documentation" / "01_Installing_Transformers.ipynb"
    assert nb.exists(), f"lesson notebook missing: {nb}"


def test_documentation_notebook_is_valid() -> None:
    """The lesson notebook must parse and validate as nbformat 4."""
    nbformat = pytest.importorskip("nbformat", reason="nbformat not installed — pip install nbformat")
    nb_path = Path(__file__).resolve().parents[1] / "documentation" / "01_Installing_Transformers.ipynb"
    notebook = nbformat.read(nb_path, as_version=4)
    nbformat.validate(notebook)
    # every cell must have an id (future-proof against stricter nbformat)
    assert all(cell.get("id") for cell in notebook.cells), "cells are missing id fields"


# --- transformers checks (skip when not installed) --------------------------

def test_transformers_installed() -> None:
    transformers = _transformers()
    assert transformers.__version__


def test_transformers_version_supported() -> None:
    transformers = _transformers()
    version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    assert version >= MIN_TRANSFORMERS, (
        f"transformers {transformers.__version__} is too old; upgrade with: pip install -U transformers"
    )


def test_transformers_pipeline_importable() -> None:
    _transformers()  # ensure installed
    from transformers import pipeline

    assert callable(pipeline)


# --- torch checks (skip when not installed) ---------------------------------

def test_torch_installed_and_functional() -> None:
    torch = _torch()
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.sum().item() == 6.0, "torch is installed but tensors do not work"


def test_cuda_check_returns_bool() -> None:
    torch = _torch()
    assert isinstance(torch.cuda.is_available(), bool)
