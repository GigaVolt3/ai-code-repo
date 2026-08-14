# HF-006 — Model Inference

Sixth lesson of **Hugging Face Fundamentals**: do inference by hand —
tokenize, forward pass, softmax, argmax — and see exactly what the pipeline
does under the hood.

## Folder layout

```
06_Model_Inference/
├── 06_Model_Inference.py        # lesson script (runnable)
├── documentation/               # lesson docs as a Jupyter notebook
│   ├── 06_Model_Inference.ipynb
│   └── README.md
├── test/                        # automated checks
│   ├── test_hf006_model_inference.py
│   └── README.md
└── resources/                   # links + minimal requirements
    ├── reference_links.md
    └── requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. the 4-step recipe on the default sentence
python 06_Model_Inference.py

# 2. your own text, with raw logits shown
python 06_Model_Inference.py --text "terrible experience" --print-logits

# 3. run the automated checks
pip install pytest
python -m pytest test/ -v

# 4. read the interactive lesson
jupyter lab documentation/06_Model_Inference.ipynb
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| the recipe | tokenize → forward → softmax → argmax |
| logits | raw scores, any sign — *not* probabilities |
| softmax | logits → probabilities, sums to 1 |
| argmax | pick the most likely class |
| pipeline equivalence | manual math and `pipeline()` give the same numbers |
| `model.eval()` / `torch.no_grad()` | inference mode: no dropout, less memory |

Uses the SST-2 sentiment model already downloaded in HF-002/05 (cached, no
new download).

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-006 | Hugging Face Fundamentals | `06_Model_Inference.py` | ☐ | ☐ | ☐ |

**Next:** HF-007 — Loading Pretrained Models.