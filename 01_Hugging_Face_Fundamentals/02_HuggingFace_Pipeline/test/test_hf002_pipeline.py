"""HF-002 — Tests for the Hugging Face Pipelines lesson.

Run:  python -m pytest test_hf002_pipeline.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

LESSON = Path(__file__).resolve().parents[1] / "02_HuggingFace_Pipeline.py"


def load_lesson():
    spec = importlib.util.spec_from_file_location("hf002", LESSON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pipeline():
    try:
        from transformers import pipeline
        return pipeline
    except ImportError:
        pytest.skip("transformers not installed")


def test_script_exists() -> None:
    assert LESSON.exists()
    text = LESSON.read_text(encoding="utf-8")
    assert "pipeline(" in text
    assert "HF-002" in text


def test_readme_exists() -> None:
    readme = LESSON.parent / "README.md"
    assert readme.exists(), "lesson README missing"


def test_notebook_exists_and_valid() -> None:
    nbformat = pytest.importorskip("nbformat")
    nb = LESSON.parent / "documentation" / "02_HuggingFace_Pipeline.ipynb"
    assert nb.exists(), f"notebook missing: {nb}"
    notebook = nbformat.read(nb, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.get("id") for cell in notebook.cells)


def test_module_imports_and_docs() -> None:
    mod = load_lesson()
    assert callable(mod.demo_sentiment)   # task demos are importable
    assert callable(mod.main)


def test_sentiment_pipeline_works() -> None:
    pipeline = _pipeline()
    sent = pipeline("sentiment-analysis")
    result = sent("I love this product!")[0]
    assert result["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= result["score"] <= 1.0


def test_classification_topk_result_shape() -> None:
    pipeline = _pipeline()
    classifier = pipeline(
        "text-classification",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
    result = classifier("It was fine, I guess.", top_k=2)
    if result and isinstance(result[0], list):  # v5 nests single texts
        result = result[0]
    assert len(result) == 2
    assert all("label" in r and "score" in r for r in result)
