# HF-201 — Text Generation

First lesson of **Hugging Face Applications**: generate text with a pretrained
causal language model (GPT-2) through the `pipeline` API.

## Folder layout

```
01_Text_Generation/
├── 01_Text_Generation.py   # lesson script (runnable)
├── documentation/          # lesson docs (teaching notebook)
│   ├── 01_Text_Generation.ipynb
│   └── README.md
├── test/                   # automated checks
│   ├── test_text_generation.py
│   └── README.md
└── resources/              # links + minimal requirements
    ├── reference_links.md
    └── requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. generate text with the default model (GPT-2)
python 01_Text_Generation.py

# 2. try your own prompt and longer output
python 01_Text_Generation.py --prompt "Once upon a time" --max 60

# 3. compare greedy vs. sampling decoding
python 01_Text_Generation.py --greedy
python 01_Text_Generation.py --temperature 0.3

# 4. run the automated checks
pip install pytest
python -m pytest test/ -v
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| `pipeline("text-generation")` | ready-made generation API, one line of code |
| `max_new_tokens` | how long the continuation may be |
| greedy decoding | deterministic — always the most probable token |
| sampling + `temperature` | random draws; low temp = focused, high temp = creative |
| `top_k` / `top_p` | restrict sampling to the most likely tokens |
| `num_return_sequences` | ask for several continuations of one prompt |

The first run downloads the model (~550 MB for `gpt2`, once, cached in
`~/.cache/huggingface`). Use `--model distilgpt2` for a much smaller one.

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-201 | Hugging Face Applications | `01_Text_Generation.py` | ☐ | ☐ | ☐ |

**Next:** HF-202 — Summarization.
