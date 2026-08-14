# HF-203 — Question Answering

Third lesson of **Hugging Face Applications**: extract answers to questions
from a passage of text using a DistilBERT model fine-tuned on SQuAD.

> **transformers v5 note:** the `pipeline("question-answering")` helper was
> removed in transformers 5.x. This lesson uses the recommended replacement:
> the model directly predicts the start/end token of the answer span. The
> script works on both v4 and v5.

## Folder layout

```
03_Question_Answering/
|-- 03_Question_Answering.py   # lesson script (runnable)
|-- documentation/             # lesson docs (teaching notebook)
|   |-- 03_Question_Answering.ipynb
|   `-- README.md
|-- test/                      # automated checks
|   |-- test_question_answering.py
|   `-- README.md
`-- resources/                 # links + minimal requirements
    |-- reference_links.md
    `-- requirements.txt
```

## Quick start

```bash
pip install -r resources/requirements.txt

# 1. ask the built-in questions about a sample passage
python 03_Question_Answering.py

# 2. ask your own question
python 03_Question_Answering.py --question "Where is the company based?"

# 3. ask about your own text
python 03_Question_Answering.py --context "The Eiffel Tower was built in 1889 in Paris."

# 4. use a stronger model
python 03_Question_Answering.py --model deepset/roberta-base-squad2

# 5. run the automated checks
pip install pytest
python -m pytest test/ -v
```

## What this lesson teaches

| Concept | Details |
|---------|---------|
| `AutoModelForQuestionAnswering` | predicts a start and an end position in the context |
| question + context | encoded together as one input |
| `start_logits` / `end_logits` | every token is scored as a possible span edge |
| `argmax` | locate the answer span, then decode it |
| extractive vs. generative QA | verbatim span from the text vs. free-form |

This start/end span search is what the removed `question-answering`
pipeline used internally.

First run downloads the model (~260 MB for the DistilBERT SQuAD model,
cached once). Switch to `--model deepset/roberta-base-squad2` for stronger
accuracy.

## Tracking

| Task ID | Module | File | Status | Tested | Documentation |
|---------|--------|------|--------|--------|---------------|
| HF-203 | Hugging Face Applications | `03_Question_Answering.py` | :white_square_button: | :white_square_button: | :white_square_button: |

**Next:** HF-204 — Translation.