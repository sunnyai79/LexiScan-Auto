# LexiScan Auto 🔍

**Intelligent Legal Contract Entity Extractor (NER)**  
*A production-grade NLP microservice for financial law firms*

---

## What It Does

LexiScan Auto automatically extracts structured entities from legal PDF contracts:

| Entity | Example |
|---|---|
| `DATE` | `January 15, 2024` → `2024-01-15` |
| `PARTY` | `Acme Corporation LLC` |
| `AMOUNT` | `$750,000.00` → `USD 750,000.00` |
| `TERMINATION_CLAUSE` | `Either party may terminate upon 30 days notice...` |

Upload a PDF, receive clean JSON — works on both **native digital** and **scanned** contracts.

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────┐
│         Week 1: OCR Pipeline        │
│  PyMuPDF (native) → Tesseract OCR   │
│  (for scanned pages)                │
│  + Image preprocessing (contrast,   │
│    sharpen, binarize)               │
│  + Text normalization               │
└────────────────┬────────────────────┘
                 │ clean text
                 ▼
┌─────────────────────────────────────┐
│      Week 1+2: NER Models           │
│                                     │
│  Primary:  Fine-tuned DistilBERT   │
│            (Week 2 transfer learning│
│             on Doccano annotations) │
│                                     │
│  Fallback: Regex rule-based         │
│            extractor                │
└────────────────┬────────────────────┘
                 │ raw entity spans
                 ▼
┌─────────────────────────────────────┐
│    Week 3: Rule-Based Validation    │
│  DATE   → must parse to real date,  │
│            normalize to YYYY-MM-DD  │
│  AMOUNT → must have currency symbol │
│            normalize to "USD X.XX"  │
│  PARTY  → must start with capital,  │
│            reject false positives   │
│  CLAUSE → must contain keywords     │
│  Deduplication of overlapping spans │
└────────────────┬────────────────────┘
                 │ validated entities
                 ▼
┌─────────────────────────────────────┐
│      Week 4: FastAPI REST API       │
│  POST /extract     ← PDF upload     │
│  POST /extract/text ← plain text   │
│  GET  /health      ← status check  │
│  Wrapped in Docker container        │
└─────────────────────────────────────┘
```

---

## Quick Start

### Option A: Docker (Recommended)

```bash
# Build and start
docker-compose up --build

# Test the API
curl http://localhost:8000/health

# Extract from text
curl -X POST http://localhost:8000/extract/text \
  -F "text=This Agreement dated January 15, 2024 between Acme LLC and Beta Corp for $500,000. Either party may terminate upon 30 days notice."

# Extract from PDF
curl -X POST http://localhost:8000/extract \
  -F "file=@contract.pdf"
```

### Option B: Local Development

```bash
# 1. Install system deps (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils

# 2. Install Python deps
pip install -r requirements.txt

# 3. Train NER model (generates synthetic data automatically)
bash scripts/train.sh

# 4. Start API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# 5. Run tests
pytest tests/ -v
```

---

## API Reference

### `POST /extract`
Upload a PDF contract for entity extraction.

**Request:** `multipart/form-data`
- `file` (required): PDF file
- `use_ocr` (optional, bool): Enable Tesseract for scanned pages (default: `true`)

**Response:**
```json
{
  "status": "success",
  "filename": "contract.pdf",
  "processing_time_seconds": 1.42,
  "word_count": 842,
  "entity_counts": {
    "DATE": 3,
    "PARTY": 2,
    "AMOUNT": 4,
    "TERMINATION_CLAUSE": 1
  },
  "entities": [
    {
      "text": "January 15, 2024",
      "label": "DATE",
      "start": 44,
      "end": 60,
      "normalized": "2024-01-15",
      "confidence": "high"
    },
    {
      "text": "$750,000.00",
      "label": "AMOUNT",
      "start": 154,
      "end": 165,
      "normalized": "USD 750,000.00",
      "confidence": "high"
    }
  ],
  "raw_text_preview": "This Agreement is entered into as of..."
}
```

### `POST /extract/text`
Extract from plain text (no PDF needed).

### `GET /health`
Check API and model status.

---

## Training with Your Own Data (Doccano)

1. **Annotate** contracts in [Doccano](https://github.com/doccano/doccano)
   - Labels: `DATE`, `PARTY`, `AMOUNT`, `TERMINATION_CLAUSE`
2. **Export** as JSONL format
3. **Train:**
```bash
python -m src.ner.training_pipeline \
  --annotation-file data/annotated/my_contracts.jsonl \
  --model-output data/models/bert_ner \
  --epochs 10
```
4. **Restart** the API — it auto-loads from `data/models/bert_ner`

**Expected F1 scores** (with 500+ annotated contracts):
- DATE: ~0.96
- AMOUNT: ~0.94
- PARTY: ~0.88
- TERMINATION_CLAUSE: ~0.82

---

## Project Structure

```
lexiscan_auto/
├── src/
│   ├── ocr/
│   │   └── ocr_pipeline.py       # Week 1: Tesseract + PyMuPDF
│   ├── ner/
│   │   ├── ner_model.py          # Week 1: BiLSTM-CRF model
│   │   ├── bert_ner.py           # Week 2: Fine-tuned BERT
│   │   ├── data_annotation.py    # Doccano format converter
│   │   └── training_pipeline.py  # Full training orchestrator
│   ├── postprocessing/
│   │   └── validator.py          # Week 3: Rule-based validation
│   └── api/
│       └── app.py                # Week 4: FastAPI REST service
├── tests/
│   ├── test_ocr.py
│   ├── test_validator.py         # 20 unit tests
│   └── test_api.py               # Integration tests
├── data/
│   ├── annotated/                # Doccano exports
│   └── models/                   # Saved model weights
├── scripts/
│   ├── train.sh
│   └── test_api.sh
├── Dockerfile                    # Week 4: Container
├── docker-compose.yml
└── requirements.txt
```

---

## Week-by-Week Implementation Summary

| Week | Component | Key Decisions |
|---|---|---|
| 1 | OCR + BiLSTM NER | PyMuPDF for digital pages; Tesseract (300 DPI, contrast/sharpen preprocessing) for scanned; BiLSTM with BIO tagging |
| 2 | Transfer Learning | DistilBERT fine-tuned on annotated data; seqeval F1 as primary metric; AdamW lr=3e-5; subword token alignment |
| 3 | Validation | Per-type regex validators; date normalization to ISO 8601; currency canonicalization; overlap deduplication |
| 4 | Deployment | FastAPI + Uvicorn; Docker with non-root user; health checks; 4GB memory limit for BERT |
