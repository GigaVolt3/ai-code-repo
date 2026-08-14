"""HF-007 — Tests for the Loading Pretrained Models lesson.

Run:  python -m pytest test_hf007_loading.py -v
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

LESSON = Path(__file__).resolve().parents[1] / "07_Loading_Pretrained_Model.py"
MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def load_lesson():
    spec = importlib.util.spec_from_file_location("hf007", LESSON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_exists() -> None:
    assert LESSON.exists()
    text = LESSON.read_text(encoding="utf-8")
    assert "from_pretrained" in text
    assert "HF-007" in text


def test_readme_exists() -> None:
    assert (LESSON.parent / "README.md").exists()


def test_notebook_exists_and_valid() -> None:
    nbformat = pytest.importorskip("nbformat")
    nb = LESSON.parent / "documentation" / "07_Loading_Pretrained_Model.ipynb"
    assert nb.exists(), f"notebook missing: {nb}"
    notebook = nbformat.read(nb, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.get("id") for cell in notebook.cells)


def test_module_imports_and_defaults() -> None:
    mod = load_lesson()
    assert mod.DEFAULT_MODEL == MODEL_ID


def test_save_reload_round_trip(tmp_path) -> None:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    reloaded = AutoModelForSequenceClassification.from_pretrained(tmp_path)
    assert reloaded.num_parameters() == model.num_parameters()


def test_revision_pinning() -> None:
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError:
        pytest.skip("transformers not installed")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision="main")
    assert model.config.num_labels == 2


def test_offline_load_from_cache() -> None:
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError:
        pytest.skip("transformers not installed")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        assert model.config.num_labels == 2
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)