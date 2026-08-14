# HF-004 — AutoTokenizer

Fourth lesson of **Hugging Face Fundamentals**: one class that loads the
*right* tokenizer for any model — BERT (WordPiece), GPT-2 (BPE), T5
(SentencePiece), and everything else.

## Folder layout

```
04_AutoTokenizer/
├── 04_AutoTokenizer.py          # lesson script (runnable)
├── documentation/               # lesson docs as a Jupyter notebook
│   ├── 04_AutoTokenizer.ipynb
│   └── README.md
├── test/                        # automated checks
│   ├── test_hf004_auto_tokenizer.py
│   └── README.md
└── resources/                   # links + minimal requirements
    ├── reference_links.md
    └── requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. default lesson (BERT tokenizer)
python 04_AutoTokenizer.py

# 2. compare BERT vs GPT-2 vs T5 tokenizer families
python 04_AutoTokenizer.py --compare

# 3. any other model id
python 04_AutoTokenizer.py --model roberta-base --text "h\u00e9llo"

# 4. save a tokenizer to a folder and reload it (temp dir demo)
python 04_AutoTokenizer.py --save-temp

# 5. run the automated checks
pip install pytest
python -m pytest test/ -v

# 6. read the interactive lesson
jupyter lab documentation/04_AutoTokenizer.ipynb
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| `AutoTokenizer.from_pretrained` | reads the config, picks the matching tokenizer |
| family differences | WordPiece `##` vs byte-level BPE vs SentencePiece `▁` |
| padding / truncation | rectangular batches, `attention_mask` marks real words |
| save / reload | tokenizers are just folders (`save_pretrained`) |

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-004 | Hugging Face Fundamentals | `04_AutoTokenizer.py` | ☐ | ☐ | ☐ |

**Next:** HF-005 — AutoModel.