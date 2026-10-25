# Email Spam Detection using BERT + LSTM / BiLSTM

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A deep-learning pipeline that fine-tunes **BERT** (`bert-base-uncased`) with a stacked **LSTM** or **Bidirectional LSTM (BiLSTM)** head to classify emails as **spam** or **ham** (legitimate). The project is implemented in a self-contained Jupyter notebook and trained on the [Enron Spam Dataset](https://www.kaggle.com/datasets/wanderfj/enron-spam).

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Running the Notebook](#running-the-notebook)
- [Key Hyperparameters](#key-hyperparameters)
- [Requirements](#requirements)

---

## Project Overview

Traditional spam filters rely on hand-crafted rules or bag-of-words models. This project takes a modern NLP approach:

1. **BERT** (`bert-base-uncased`) encodes each email's full token sequence into contextual embeddings.
2. An **LSTM** or **BiLSTM** layer processes the BERT sequence output, capturing sequential dependencies with packed-sequence support to correctly handle padding.
3. A linear **classifier head** maps the recurrent state to spam/ham logits.
4. Both models are compared on held-out test metrics (accuracy, precision, recall, F1).

---

## Dataset

| Property | Value |
|---|---|
| Source | Enron Spam Dataset (`enron_spam_data.csv`) |
| Total rows (raw) | 33,716 |
| Usable rows (after cleaning) | ~30,137 |
| Classes | `ham` (0) · `spam` (1) |
| Class balance | ~52% ham · ~48% spam |
| Train / Val / Test split | 70% / 15% / 15% (stratified) |

> **Note:** The dataset file (`enron_spam_data.csv`) is **not** included in this repository due to its size. Download it from [Kaggle](https://www.kaggle.com/datasets/wanderfj/enron-spam) and place it in the project root, or upload it when prompted inside the notebook (Colab file-upload widget).

The notebook combines the **Subject** and **Message** columns into a single `email_text` field and maps labels to binary integers (ham → 0, spam → 1).

---

## Architecture

```
Email text
    │
    ▼
BERT tokenizer  (max_length = 128, padding, truncation)
    │
    ▼
bert-base-uncased  (12 layers, hidden_size = 768)
    │  last_hidden_state  [batch, seq_len, 768]
    ▼
Pack padded sequence
    │
    ▼
┌─────────────────────────────────┐
│  LSTM  (hidden=128, 1 layer)    │   ← BERT + LSTM variant
│    or                           │
│  BiLSTM (hidden=128, 1 layer,   │   ← BERT + BiLSTM variant
│          bidirectional=True)    │
└─────────────────────────────────┘
    │  final hidden state(s)
    ▼
Dropout (p = 0.3)
    │
    ▼
Linear  (128 or 256 → 2)
    │
    ▼
CrossEntropyLoss / argmax → spam / ham
```

---

## Results

Results from 2 training epochs on a **Tesla T4 GPU** (Google Colab):

| Model | Val Accuracy | Val F1 | Test ready |
|---|---|---|---|
| BERT + LSTM | 99.09% | 0.9906 | ✅ |
| BERT + BiLSTM | 99.23% | 0.9920 | ✅ |

Training history (BERT + LSTM):

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Val F1 |
|---|---|---|---|---|---|
| 1 | 0.1135 | 0.0639 | 96.20% | 98.47% | 0.9840 |
| 2 | 0.0176 | 0.0423 | 99.66% | 99.09% | 0.9906 |

Training history (BERT + BiLSTM):

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Val F1 |
|---|---|---|---|---|---|
| 1 | 0.1123 | 0.0317 | — | 99.18% | 0.9915 |
| 2 | 0.0155 | 0.0405 | — | 99.23% | 0.9920 |

---

## Repository Structure

```
email-spam-detection/
├── email_spam_detection.ipynb   # Main notebook (complete pipeline)
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Kesanisaicharan/email-spam-detection.git
cd email-spam-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU highly recommended.** BERT fine-tuning on ~21k samples takes ~10 min/epoch on a T4 GPU. CPU training will be much slower.

### 4. Obtain the dataset

Download `enron_spam_data.csv` from [Kaggle – Enron Spam](https://www.kaggle.com/datasets/wanderfj/enron-spam) and place it in the project root:

```
email-spam-detection/
└── enron_spam_data.csv   ← put it here
```

---

## Running the Notebook

### Option A – Google Colab (recommended for GPU)

1. Open `email_spam_detection.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Set the runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU).
3. Run all cells in order. The first data-loading cell shows a file-upload widget — upload `enron_spam_data.csv` there.

### Option B – Local Jupyter

```bash
jupyter notebook email_spam_detection.ipynb
```

Make sure `enron_spam_data.csv` is in the same directory as the notebook (the local-fallback path in the notebook is `"messages.csv"` — rename or update the path if needed).

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Max token length | 128 |
| Batch size | 8 |
| Epochs | 2 |
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| LR scheduler | Linear warmup (10%) |
| Gradient clipping | 1.0 |
| LSTM hidden size | 128 |
| Dropout | 0.3 |
| Random seed | 42 |

---

## Requirements

See [requirements.txt](requirements.txt) for the full list. Key packages:

| Package | Purpose |
|---|---|
| `torch` | Deep learning framework |
| `transformers` | BERT model & tokenizer |
| `accelerate` | Mixed-precision / device helpers |
| `scikit-learn` | Train/val/test split, metrics |
| `pandas` / `numpy` | Data wrangling |
| `matplotlib` / `seaborn` | Visualisation & confusion matrix |
| `ipywidgets` | Colab file-upload widget |

---

## License

This project is released under the [MIT License](LICENSE).