"""
HF-001 — Installing Hugging Face Transformers
=============================================

First lesson of the course. It checks that your machine is ready to run
Hugging Face models, explains why each component matters, and can even
install the missing pieces for you.

What we check:

    1. Python version      -> transformers requires Python 3.8+
    2. transformers        -> the library used in every lesson of this course
    3. torch (PyTorch)     -> the engine that actually runs the models
    4. sentencepiece       -> optional helper for some tokenizers (T5, Llama, ...)

One-click usage (recommended for beginners):

    python 01_Installing_Transformers.py --install --smoke
    # ^ installs anything that is missing, then runs a mini end-to-end test

Manual installation (if you prefer to install yourself):

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
    pip install transformers torch

    # or install everything this course uses:
    pip install -r requirements.txt

Usage:

    python 01_Installing_Transformers.py             # check only
    python 01_Installing_Transformers.py --install   # install missing pieces
    python 01_Installing_Transformers.py --smoke     # also run an end-to-end test
"""

from __future__ import annotations

import platform
import subprocess
import sys

# Packages checked by this script (module name == package name for all of these).
REQUIRED = ["transformers", "torch"]  # needed for every lesson
OPTIONAL = ["sentencepiece"]          # needed by some tokenizers only


def intro() -> None:
    """Explain to a beginner what this script does and why."""
    print("=" * 60)
    print("HF-001 — Installing Hugging Face Transformers")
    print("=" * 60)
    print("This script checks that your machine is ready to run Hugging Face models.")
    print("It verifies 4 things and explains why each one matters:")
    print("  1. Python version  -> transformers requires Python 3.8+")
    print("  2. transformers    -> the library we use in every lesson of this course")
    print("  3. torch (PyTorch) -> the engine that actually runs the models")
    print("  4. sentencepiece   -> optional helper for some tokenizers (T5, Llama, ...)")
    print()


def check_python() -> bool:
    """Transformers requires Python 3.8+."""
    ok = sys.version_info >= (3, 8)
    print(f"[python]       {platform.python_version()}  {'OK' if ok else 'TOO OLD (3.8+ required)'}")
    return ok


def check_transformers() -> bool:
    """Import transformers and report its version."""
    try:
        import transformers
    except ImportError:
        print("[transformers] NOT installed — run:  pip install transformers")
        return False
    print(f"[transformers] {transformers.__version__}  OK")
    return True


def check_torch() -> bool:
    """PyTorch is the default execution backend. Report version and hardware."""
    try:
        import torch
    except ImportError:
        print("[torch]        NOT installed — run:  pip install torch")
        return False
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "CPU"
    print(f"[torch]        {torch.__version__}  OK")
    print(f"[cuda]         available={cuda}  device={device}")
    return True


def check_sentencepiece() -> bool:
    """Optional: needed by many multilingual / BPE tokenizers (T5, Llama, ...)."""
    try:
        import sentencepiece
    except ImportError:
        print("[sentencepiece] (optional) NOT installed — some tokenizers will not load.")
        return False
    print(f"[sentencepiece] {sentencepiece.__version__}  OK")
    return True


def missing_packages() -> list[str]:
    """Return the names of required/optional packages that are not installed."""
    missing = []
    for pkg in REQUIRED + OPTIONAL:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def install_missing(packages: list[str]) -> None:
    """One-click helper: pip-install the given packages into the current Python."""
    print(f"\nInstalling missing packages: {', '.join(packages)}")
    print("(this may take a minute or two on the first run)")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", *packages], check=True)
        print("Install finished. Re-checking the environment ...\n")
    except subprocess.CalledProcessError:
        print("Installation failed — run the install command manually to see the full error.")
        sys.exit(1)


def smoke_test() -> None:
    """Download a small model and run one prediction to prove the install end-to-end.

    Uses the canonical 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
    (~270 MB, downloaded once and cached) — small enough to run on CPU and known
    to be compatible with current transformers versions.
    """
    print("\nRunning a smoke test with a small sentiment model ...")
    try:
        from transformers import pipeline
    except ImportError:
        print("Smoke test skipped: transformers is not installed.")
        return

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
    result = classifier("The Hugging Face ecosystem is awesome!")[0]
    print(f"  '{result['label']}' with confidence {result['score']:.3f}")
    print("Smoke test passed — the installation works end to end.")


def main() -> None:
    intro()

    results = [
        check_python(),
        check_transformers(),
        check_torch(),
        check_sentencepiece(),
    ]

    # One-click mode: install anything that is missing, then re-check.
    if "--install" in sys.argv:
        missing = missing_packages()
        if missing:
            install_missing(missing)
            results = [
                check_python(),
                check_transformers(),
                check_torch(),
                check_sentencepiece(),
            ]
        else:
            print("Everything is already installed — nothing to do.")

    if "--smoke" in sys.argv:
        smoke_test()

    print("=" * 60)
    if all(results):
        print("All good — your environment is ready for HF-002 (pipelines).")
    else:
        print("Some components are missing. Either re-run with --install,")
        print("or follow the steps in documentation/01_Installing_Transformers.ipynb.")
        sys.exit(1)


if __name__ == "__main__":
    main()
