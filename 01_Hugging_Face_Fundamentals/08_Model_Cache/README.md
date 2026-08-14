# HF-008 — Model Cache

Eighth and last lesson of **Hugging Face Fundamentals**: where downloaded
models live on disk, how to measure them, and how to free space.

## Folder layout

```
08_Model_Cache/
├── 08_Model_Cache.py            # lesson script (runnable)
├── documentation/               # lesson docs as a Jupyter notebook
│   ├── 08_Model_Cache.ipynb
│   └── README.md
├── test/                        # automated checks
│   ├── test_hf008_model_cache.py
│   └── README.md
└── resources/                   # links + minimal requirements
    ├── reference_links.md
    └── requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. list everything cached, biggest first (default action)
python 08_Model_Cache.py --list

# 2. where the cache lives + env variables
python 08_Model_Cache.py --details

# 3. total disk usage
python 08_Model_Cache.py --du

# 4. delete one cached model (it re-downloads later on demand)
python 08_Model_Cache.py --remove distilgpt2

# 5. run the automated checks
pip install pytest
python -m pytest test/ -v

# 6. read the interactive lesson
jupyter lab documentation/08_Model_Cache.ipynb
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| cache location | `~/.cache/huggingface/hub`, override with `HF_HOME` |
| layout | `models--OWNER--NAME` folders, `snapshots/` + `blobs/` |
| `scan_cache_dir` | official API: models, revisions, sizes |
| `delete_revisions` | free disk space safely (re-downloads on demand) |
| offline mode | `HF_HUB_OFFLINE=1` proves what is cached |

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-008 | Hugging Face Fundamentals | `08_Model_Cache.py` | ☐ | ☐ | ☐ |

This is the end of the module. Next: **HF-101** — Hugging Face Applications.