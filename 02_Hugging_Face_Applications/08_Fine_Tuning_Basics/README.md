# HF-208 — Fine-Tuning Basics

Eighth lesson of **Hugging Face Applications**: adapt a pretrained DistilBERT
model to your own task with a small labeled dataset and the `Trainer` API.

## Folder layout

```
08_Fine_Tuning_Basics/
|-- 08_Fine_Tuning_Basics.py   # lesson script (runnable)
|-- documentation/             # lesson docs (teaching notebook)
|   |-- 08_Fine_Tuning_Basics.ipynb
|   `-- README.md
|-- test/                      # automated checks
|   |-- test_fine_tuning_basics.py
|   `-- README.md
`-- resources/                 # links + minimal requirements
    |-- reference_links.md
    `-- requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. run the full fine-tuning demo (CPU-friendly, ~1-2 min)
python 08_Fine_Tuning_Basics.py

# 2. train longer
python 08_Fine_Tuning_Basics.py --epochs 8

# 3. save to a different folder
python 08_Fine_Tuning_Basics.py --save-dir models/my-tuned-model

# 4. run the automated checks
pip install pytest
python -m pytest test/ -v
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| `AutoModelForSequenceClassification` | pretrained backbone + new classification head |
| `Trainer` / `TrainingArguments` | the standard fine-tuning loop |
| tokenization function | turn texts into `input_ids` + `attention_mask` |
| `save_model()` / `from_pretrained()` | persist the tuned model and reload it |
| before vs. after | accuracy on our eval set rises after tuning |

The dataset is built into the script (no `datasets` dependency, no extra
downloads). First run downloads `distilbert-base-uncased` (~270 MB, cached).

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-208 | Hugging Face Applications | `08_Fine_Tuning_Basics.py` | :white_square_button: | :white_square_button: | :white_square_button: |

**This is the last lesson of Hugging Face Applications.**
