# HF-007 — Loading Pretrained Models

Seventh lesson of **Hugging Face Fundamentals**: one API
(`from_pretrained`) for every source — the Hub, a local folder, a pinned
revision — plus offline mode for machines without internet.

## Folder layout

```
07_Loading_Pretrained_Model/
├── 07_Loading_Pretrained_Model.py   # lesson script (runnable)
├── documentation/                   # lesson docs as a Jupyter notebook
│   ├── 07_Loading_Pretrained_Model.ipynb
│   └── README.md
├── test/                            # automated checks
│   ├── test_hf007_loading.py
│   └── README.md
└── resources/                       # links + minimal requirements
    ├── reference_links.md
    └── requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. Hub load, local-folder snapshot, save/reload, revisions, offline mode
python 07_Loading_Pretrained_Model.py

# 2. pin a specific revision
python 07_Loading_Pretrained_Model.py --revision main

# 3. cache-only mode (proves the model is cached)
python 07_Loading_Pretrained_Model.py --offline

# 4. run the automated checks
pip install pytest
python -m pytest test/ -v

# 5. read the interactive lesson
jupyter lab documentation/07_Loading_Pretrained_Model.ipynb
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| sources | Hub id, local folder, revision — same `from_pretrained` |
| `snapshot_download` | copy a model to a folder once (`ignore_patterns` trims it) |
| save / reload | the exact workflow fine-tuning (HF-208) ends with |
| revisions | branches, tags, commit hashes — reproducible weights |
| offline mode | `HF_HUB_OFFLINE=1` forces cache-only loading |

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-007 | Hugging Face Fundamentals | `07_Loading_Pretrained_Model.py` | ☐ | ☐ | ☐ |

**Next:** HF-008 — Model Cache.