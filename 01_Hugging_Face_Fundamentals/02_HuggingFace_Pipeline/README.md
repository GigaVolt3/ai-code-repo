# HF-002 — Hugging Face Pipelines

Second lesson of **Hugging Face Fundamentals**: the `pipeline` — one line of
code that loads a model, its tokenizer and the post-processing, all at once.

## Folder layout

```
02_HuggingFace_Pipeline/
├── 02_HuggingFace_Pipeline.py   # lesson script (runnable)
├── documentation/               # lesson docs as a Jupyter notebook
│   ├── 02_HuggingFace_Pipeline.ipynb
│   └── README.md
├── test/                        # automated checks
│   ├── test_hf002_pipeline.py
│   └── README.md
└── resources/                   # links + minimal requirements
    ├── reference_links.md
    └── requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. tour every pipeline task (sentiment, classification, zero-shot, generation)
python 02_HuggingFace_Pipeline.py

# 2. run one task only
python 02_HuggingFace_Pipeline.py --task zero-shot

# 3. run the automated checks
pip install pytest
python -m pytest test/ -v

# 4. read the interactive lesson
jupyter lab documentation/02_HuggingFace_Pipeline.ipynb
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| `pipeline(task, model)` | tokenizer + model + post-processing in one object |
| hidden recipe | tokenize → forward pass → post-process |
| batching | pass a list of texts; results come back in order |
| `device` | `-1` = CPU (safe default), `0` = first GPU |
| task zoo | sentiment-analysis, text-classification, zero-shot, text-generation |

Models are downloaded on first use (~250–550 MB each, once) and then cached
on disk (see HF-008 for how to manage the cache).

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-002 | Hugging Face Fundamentals | `02_HuggingFace_Pipeline.py` | ☐ | ☐ | ☐ |

**Next:** HF-003 — Tokenizer.