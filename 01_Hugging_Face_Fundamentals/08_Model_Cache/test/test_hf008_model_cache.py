"""HF-008 — Tests for the Model Cache lesson.

Run:  python -m pytest test_hf008_model_cache.py -v
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

LESSON = Path(__file__).resolve().parents[1] / "08_Model_Cache.py"


def load_lesson():
    spec = importlib.util.spec_from_file_location("hf008", LESSON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_exists() -> None:
    assert LESSON.exists()
    text = LESSON.read_text(encoding="utf-8")
    assert "scan_cache_dir" in text
    assert "HF-008" in text


def test_readme_exists() -> None:
    assert (LESSON.parent / "README.md").exists()


def test_notebook_exists_and_valid() -> None:
    nbformat = pytest.importorskip("nbformat")
    nb = LESSON.parent / "documentation" / "08_Model_Cache.ipynb"
    assert nb.exists(), f"notebook missing: {nb}"
    notebook = nbformat.read(nb, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.get("id") for cell in notebook.cells)


def test_module_imports_and_helpers() -> None:
    mod = load_lesson()
    assert mod.human(1024) == "1.0 KB"
    assert mod.human(2 * 1024 * 1024) == "2.0 MB"
    assert mod.cache_root is not None


def test_cache_root_prefers_hf_home(monkeypatch) -> None:
    mod = load_lesson()
    monkeypatch.setenv("HF_HOME", r"D:\custom\hf")
    assert mod.cache_root() == Path(r"D:\custom\hf\hub")


def test_scan_cache_dir_api() -> None:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        pytest.skip("huggingface_hub not installed")
    m = load_lesson()
    info = scan_cache_dir(cache_dir=m.cache_root())
    assert len(info.repos) >= 0
    assert info.size_on_disk >= 0
    for repo in info.repos:
        assert repo.repo_id
        assert repo.size_on_disk >= 0


def test_scan_uses_repo_size_str() -> None:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        pytest.skip("huggingface_hub not installed")
    m = load_lesson()
    info = scan_cache_dir(cache_dir=m.cache_root())
    for repo in info.repos:
        if repo.size_on_disk_str:
            break
    assert repo.size_on_disk_str  # API used by the lesson script exists