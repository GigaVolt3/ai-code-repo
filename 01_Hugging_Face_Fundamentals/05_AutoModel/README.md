# HF-005 — AutoModel

Fifth lesson of **Hugging Face Fundamentals**: load any pretrained
architecture by id, inspect its blueprint, and run one forward pass.

## Folder layout

```
05_AutoModel/
├── 05_AutoModel.py              # lesson script (runnable)
├── documentation/              # lesson docs as a Jupyter notebook
│   ├── 05_AutoModel.ipynb
│   └── README.md
├── test/                       # automated checks
│   ├── test_hf005_auto_model.py
│   └── README.md
└── resources/                  # links + minimal requirements
    ├── reference_links.md
    └── requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. load DistilBERT, inspect config, run one forward pass
python 05_AutoModel.py

# 2. the bigger sibling
python 05_AutoModel.py --model bert-base-uncased

# 3. plain AutoModel without a task head
python 05_AutoModel.py --no-head

# 4. run the automated checks
pip install pytest
python -m pytest test/ -v

# 5. read the interactive lesson
jupyter lab documentation/05_AutoModel.ipynb
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| `AutoModel.from_pretrained` | config → architecture → weights, automatically |
| the blueprint | `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `vocab_size` |
| forward pass | output shape `(batch, tokens, hidden_size)` |
| task heads | `AutoModelForSequenceClassification` etc. turn vectors into scores |
| parameter counting | DistilBERT ≈ 66 M, BERT ≈ 110 M (and why small = fast on CPU) |

First run downloads `distilbert-base-uncased` (~270 MB, cached after that).

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-005 | Hugging Face Fundamentals | `05_AutoModel.py` | ☐ | ☐ | ☐ |

**Next:** HF-006 — Model Inference.