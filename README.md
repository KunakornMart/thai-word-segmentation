# Super AI Engineer S6 – Thai Word Segmentation Challenge

> Character-level Thai word segmentation solution for **Super AI Engineer Season 6 – Word Segmentation Challenge**.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-BiLSTM-EE4C2C?logo=pytorch&logoColor=white)
![Thai NLP](https://img.shields.io/badge/Thai%20NLP-Word%20Segmentation-2EC866)
![LST20](https://img.shields.io/badge/Dataset-LST20-529EFF)
![Score](https://img.shields.io/badge/Private%20Score-0.97628-EDB227)
![Rank](https://img.shields.io/badge/Private%20Rank-2-purple)

---

## Highlights

| Item | Result |
|---|---:|
| Competition | **Super AI Engineer Season 6 – Word Segmentation Challenge** |
| Task | Thai word segmentation / character-level sequence labeling |
| Private Leaderboard Rank | **2** |
| Private Score | **0.97628** |
| Team | `600637-คุณากร` |
| Evaluation | Accuracy-style score on BIO word-boundary labels |
| Competition Link | [Kaggle Leaderboard](https://www.kaggle.com/competitions/super-ai-engineer-ss-6-word-segmentation/leaderboard) |

---

## Project Overview

This project solves a Thai word segmentation challenge using character-level sequence modeling.

The task is to predict a segmentation label for each character in the test text.

Possible prediction labels:

| Label | Meaning |
|---|---|
| `B_WORD` | Beginning of a word |
| `I_WORD` | Inside a word |
| `E_WORD` | End of a word |

The competition provides a raw Thai test text and requires a submission file with one prediction for each character ID.

---

## Competition Summary

| Item | Detail |
|---|---|
| Start | Apr 3, 2026 |
| Close | Apr 4, 2026 |
| Host | Super AI Engineer |
| Platform | Kaggle |
| Public Test | 50% |
| Private Test | 50% |
| Daily Submission Limit | 5 submissions/day |
| Submission Rows | 35,182 |
| Test Text Length | 37,248 characters |
| Dataset License | CC BY-NC-SA 4.0 |

Competition page:  
https://www.kaggle.com/competitions/super-ai-engineer-ss-6-word-segmentation

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── LST20 Annotation Guideline.pdf
│   ├── LST20 Brief Specification.pdf
│   ├── ws_list.txt
│   ├── ws_sample_submission.csv
│   └── ws_test.txt
├── notebooks/
│   └── thai_word_segmentation_bilstm_solution.ipynb
├── src/
│   ├── bio_utils.py
│   ├── validate_submission.py
│   └── wordseg_bilstm_pipeline.py
├── docs/
│   └── linkedin_project_entry.md
└── results/
    └── README.md
```

---

## Dataset

The competition package contains:

| File | Description |
|---|---|
| `ws_test.txt` | Raw Thai test text |
| `ws_list.txt` | List of possible prediction labels |
| `ws_sample_submission.csv` | Required submission format with 35,182 rows |
| `LST20 Annotation Guideline.pdf` | Annotation guideline for LST20 |
| `LST20 Brief Specification.pdf` | Brief specification of the LST20 corpus |

The training corpus is based on **LST20**, which can be downloaded separately from Hugging Face:

https://huggingface.co/datasets/lst-nectec/lst20

The full LST20 training corpus is not included in this repository.  
Only the competition files are included.

---

## Task Formulation

Instead of directly predicting `B_WORD`, `I_WORD`, and `E_WORD`, this solution formulates the task as **word-boundary detection**.

For each character, the model predicts:

```text
1 = this character ends a word
0 = this character does not end a word
```

The predicted boundaries are then converted into BIO-style word labels:

```text
Boundary predictions → Word spans → B_WORD / I_WORD / E_WORD labels
```

This formulation makes training more stable because the model only needs to learn where words end.

---

## Methodology

The pipeline follows this flow:

```text
LST20 training corpus
   ↓
Parse word-level annotations
   ↓
Convert words into character-level boundary labels
   ↓
Build character vocabulary
   ↓
Encode character type features
   ↓
Train BiLSTM boundary model
   ↓
Tune boundary threshold
   ↓
Retrain on train + validation data
   ↓
Predict test boundaries
   ↓
Decode boundaries into B/I/E labels
   ↓
Generate submission.csv
```

---

## Model Architecture

The solution uses a character-level BiLSTM model.

Main components:

- Character embedding
- Character-type embedding
- Bidirectional LSTM encoder
- Layer normalization
- Feed-forward classification head
- Binary boundary prediction

### Character Types

Each character is assigned a simple type feature:

| Type | Meaning |
|---|---|
| Thai base character |
| Thai combining mark |
| Digit |
| Latin character |
| Punctuation / symbol |
| Other |

This helps the model handle Thai characters, marks, digits, and punctuation differently.

---

## Training Strategy

The model is trained with:

- AdamW optimizer
- Binary cross-entropy loss
- Mixed precision training when GPU is available
- Gradient clipping
- Validation-based threshold tuning
- Early stopping
- Final retraining on train + validation splits

The notebook exports multiple submission variants for single-character word handling.

---

## Decoding Logic

Boundary predictions are converted into final labels as follows:

| Word length | Output labels |
|---:|---|
| 1 | `B_WORD` or configured single-character label |
| 2 | `B_WORD`, `E_WORD` |
| 3+ | `B_WORD`, `I_WORD`, ..., `E_WORD` |

The final character of each segment is forced to be an end-of-word boundary to prevent invalid decoding.

---

## Submission Format

The official submission requires two columns:

```csv
Id,Predicted
0,B_WORD
1,I_WORD
2,E_WORD
...
```

Rules:

- Exactly 35,182 rows
- Column names must be `Id` and `Predicted`
- `Predicted` must be one of:
  - `B_WORD`
  - `I_WORD`
  - `E_WORD`
- No missing IDs
- No duplicated IDs

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/KunakornMart/thai-word-segmentation.git
cd thai-word-segmentation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download LST20

Download the LST20 corpus from Hugging Face:

https://huggingface.co/datasets/lst-nectec/lst20

Place it locally and update the notebook path:

```python
LST20_ROOT = "/path/to/Corpus-LST20"
COMP_ROOT = "./data"
OUT_DIR = "./results"
```

### 4. Run the Notebook

Open:

```text
notebooks/thai_word_segmentation_bilstm_solution.ipynb
```

Run all cells to train the model and generate submission files.

### 5. Validate Submission

```bash
python src/validate_submission.py results/submission.csv --sample data/ws_sample_submission.csv
```

---

## Utility Scripts

### `src/bio_utils.py`

Contains helpers for:

- Word boundary to BIO label conversion
- Submission label validation
- Thai character type detection

### `src/validate_submission.py`

Checks whether a submission follows the official format:

- Correct columns
- Correct row count
- No missing predictions
- Valid label set
- IDs aligned with the sample submission

### `src/wordseg_bilstm_pipeline.py`

A cleaned pipeline skeleton summarizing the notebook workflow for portfolio readability.

---

## Skills Demonstrated

- Thai Natural Language Processing
- Word Segmentation
- Sequence Labeling
- Character-level Modeling
- BiLSTM
- PyTorch
- LST20 Corpus Processing
- BIO Tag Decoding
- Threshold Tuning
- Kaggle Competition Workflow
- Submission Validation

---

## Key Takeaways

This challenge demonstrates how Thai word segmentation can be framed as a character-level boundary detection problem.

Key lessons:

- Boundary prediction is a clean formulation for word segmentation
- Character-type features help with Thai-specific text patterns
- Threshold tuning can significantly affect BIO decoding quality
- Proper validation is essential for competition submissions
- A simple, well-tuned BiLSTM can perform strongly on Thai segmentation tasks

---

## LinkedIn Project Description

Built a character-level Thai word segmentation model for the Super AI Engineer Season 6 Word Segmentation Challenge. The solution uses LST20 training data, character and character-type embeddings, a BiLSTM boundary detector, threshold tuning, and BIO label decoding to generate valid `B_WORD`, `I_WORD`, and `E_WORD` predictions.

Achieved **Private Leaderboard Rank 2** with a **0.97628** private score.

---

## Author

**Kunakorn Pruksakorn**  
Automation Engineer · Data Science · AI / NLP · Industrial IoT

- GitHub: [KunakornMart](https://github.com/KunakornMart)
- Portfolio: [kunakornmart.github.io](https://kunakornmart.github.io)
- LinkedIn: [Kunakorn Pruksakorn](https://linkedin.com/in/kunakorn-pruksakorn)

---

## License

This repository is provided for portfolio and educational purposes.

Competition data follows the original dataset license: **CC BY-NC-SA 4.0**.  
Code in this repository is released under the MIT License unless otherwise noted.
