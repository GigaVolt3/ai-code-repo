"""
HF-206 — Tests for the Sentiment Analysis lesson.

Run from this folder:

    python -m pytest test_sentiment_analysis.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

LESSON_DIR = Path(__file__).resolve().parents[1]
SCRIPT = LESSON_DIR / "06_Sentiment_Analysis.py"


def test_lesson_script_exists() -> None:
    assert SCRIPT.exists(), f"lesson script missing: {SCRIPT}"
    text = SCRIPT.read_text(encoding="utf-8")
    assert "sentiment-analysis" in text
    assert "label" in text
    assert "score" in text


def test_script_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    compile(source, str(SCRIPT), "exec")


def test_documentation_notebook_exists() -> None:
    nb = LESSON_DIR / "documentation" / "06_Sentiment_Analysis.ipynb"
    assert nb.exists(), f"lesson notebook missing: {nb}"


def test_documentation_notebook_is_valid() -> None:
    """The lesson notebook must parse and validate as nbformat 4."""
    nbformat = pytest.importorskip("nbformat", reason="nbformat not installed — pip install nbformat")
    nb_path = LESSON_DIR / "documentation" / "06_Sentiment_Analysis.ipynb"
    notebook = nbformat.read(nb_path, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.get("id") for cell in notebook.cells), "cells are missing id fields"


def test_transformers_pipeline_importable() -> None:
    pytest.importorskip("transformers", reason="transformers not installed")
    from transformers import pipeline

    assert callable(pipeline)


def test_sentiment_analysis_pipeline_available() -> None:
    transformers = pytest.importorskip("transformers", reason="transformers not installed")
    tasks = set(transformers.pipelines.SUPPORTED_TASKS)
    tasks.update(getattr(transformers.pipelines, "TASK_ALIASES", {}))
    assert "sentiment-analysis" in tasks
