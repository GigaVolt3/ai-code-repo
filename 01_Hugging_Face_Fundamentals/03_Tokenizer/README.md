# HF-003 — Tokenizer

Third lesson of **Hugging Face Fundamentals**: the bridge between words and
numbers. Split text into tokens, map tokens to ids, and back.

## Folder layout

```
03_Tokenizer/
├── 03_Tokenizer.py             # lesson script (runnable)
├── documentation/              # lesson docs as a Jupyter notebook
│   ├── 03_Tokenizer.ipynb
│   └── README.md
├── test/                       # automated checks
│   ├── test_hf003_tokenizer.py
│   └── README.md
└── resources/                  # links + minimal requirements
    ├── reference_links.md
    └── requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. tokenize the default sentence, step by step
python 03_Tokenizer.py

# 2. tokenize your own text and peek at the vocabulary
python 03_Tokenizer.py --text "unhappiness!" --show-vocab 10

# 3. just the special tokens ([CLS], [SEP], [PAD], [UNK])
python 03_Tokenizer.py --special

# 4. run the automated checks
pip install pytest
python -m pytest test/ -v

# 5. read the interactive lesson
jupyter lab documentation/03_Tokenizer.ipynb
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| tokenize / decode | text → tokens → ids, and back |
| WordPiece subwords | rare words split into `##`-pieces (`unhappiness` → `un` `##ha` ...) |
| special tokens | `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]` and their ids |
| `[UNK]` | anything outside the ~30k vocab is replaced (info loss!) |
| batching | `padding=True`, `truncation=True`, rectangular batches |

Only the tokenizer files are downloaded (~1 MB) — no model weights needed.

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-003 | Hugging Face Fundamentals | `03_Tokenizer.py` | ☐ | ☐ | ☐ |

**Next:** HF-004 — AutoTokenizer.