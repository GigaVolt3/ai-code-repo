# HF-001 — Installing Transformers

First lesson of **Hugging Face Fundamentals**: install the 🤗 Transformers
library and verify your environment.

## Folder layout

```
01_Installing_Transformers/
├── 01_Installing_Transformers.py   # lesson script (runnable)
├── documentation/                  # lesson docs as a Jupyter notebook
│   ├── 01_Installing_Transformers.ipynb
│   └── README.md
├── test/                           # automated environment checks
│   ├── test_installing_transformers.py
│   └── README.md
├── resources/                      # links + minimal requirements
│   ├── reference_links.md
│   └── requirements.txt
└── screenshots/                    # evidence screenshots (see README inside)
```

## Quick start

**One click (recommended — installs anything missing, then tests it):**

```bash
python 01_Installing_Transformers.py --install --smoke
```

**Step by step:**

```bash
pip install -r resources/requirements.txt

# 1. verify the environment
python 01_Installing_Transformers.py

# 2. run the mini end-to-end smoke test
python 01_Installing_Transformers.py --smoke

# 3. run the automated checks
pip install pytest
python -m pytest test/ -v

# 4. read the interactive lesson
jupyter lab documentation/01_Installing_Transformers.ipynb
```

Script flags: `--install` pip-installs missing packages, `--smoke` runs an
end-to-end model test. Combine them for a fully automatic setup.

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-001 | Hugging Face Fundamentals | `01_Installing_Transformers.py` | ☐ | ☐ | ☐ |

**Next:** HF-002 — Hugging Face Pipelines.
