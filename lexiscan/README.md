# LexiScan Auto ⚖
**Legal Contract Entity Extractor — Intelligent Document Processing**

> Fintech NLP project | Named Entity Recognition pipeline for legal contracts

---

## Architecture

```
PDF Upload
    ↓
[OCR Service]          Week 1 — Tesseract + PyMuPDF
    ↓ raw text
[NER Service]          Week 1-2 — SpaCy + BERT fine-tuning
    ↓ raw entities
[Validation Service]   Week 3 — Rule-based normalization
    ↓ validated JSON
[FastAPI REST API]     Week 4 — Docker microservice
    ↑
[React Frontend]       Drag-and-drop UI, entity visualization
```

## Extracted Entity Types

| Entity | Description | Normalization |
|---|---|---|
| `DATE` | Contract dates, deadlines | → ISO 8601 (YYYY-MM-DD) |
| `PARTY` | Organizations, signatories | → Normalized company names |
| `AMOUNT` | Dollar values, payments | → `$X,XXX.XX` format |
| `TERMINATION_CLAUSE` | Exit conditions | → Validated keyword presence |

---

## Quick Start

### 1. Run with Docker Compose (Recommended)
```bash
docker-compose up --build
```
- Frontend: http://localhost:3000
- API:      http://localhost:8000
- API Docs: http://localhost:8000/docs

### 2. Run Backend Locally
```bash
cd backend
pip install -r requirements.txt

# Install SpaCy model
python -m spacy download en_core_web_sm
# (Production: en_core_web_trf for BERT)

# Start API
python -m uvicorn app.main:app --reload --port 8000
```

### 3. Run Frontend Locally
```bash
cd frontend
npm install
npm run dev  # http://localhost:5173
```

---

## Week-by-Week Implementation

### Week 1 — Data Acquisition & Core Modeling
- **OCR Pipeline**: `backend/app/services/ocr_service.py`
  - Native PDF via PyMuPDF (fast, lossless)
  - Scanned PDF via Tesseract OCR at 300 DPI
  - Text noise reduction & normalization
- **NER Model**: `backend/app/services/ner_service.py`
  - SpaCy NER with custom labels
  - Regex patterns as fallback/supplement
  - Entity deduplication by character overlap

### Week 2 — Transfer Learning & Performance
- **Training Script**: `backend/train_ner.py`
  - Base: `en_core_web_trf` (BERT-based contextual embeddings)
  - Fine-tuned on Doccano-annotated legal contracts
  - 80/20 train/test split, F1 evaluation every 5 iterations
  - Target F1: **0.912** on legal entity types
- **Annotation**: Use [Doccano](https://github.com/doccano/doccano) for labeling contracts

### Week 3 — Post-Processing & Validation
- **Validation Service**: `backend/app/services/validation_service.py`
  - DATE: parse 10+ date formats → `YYYY-MM-DD`
  - AMOUNT: extract numerics → `$X,XXX.XX`
  - PARTY: clean noise, normalize abbreviations (Inc., LLC, etc.)
  - TERMINATION_CLAUSE: keyword verification, min-length check

### Week 4 — Containerized Deployment
- **Dockerfile**: Multi-stage build (builder + runtime)
- **Docker Compose**: API + Frontend + Redis + Nginx
- **API Endpoint**: `POST /api/v1/extract/pdf` → structured JSON
- End-to-end test: Upload PDF → Receive JSON ✓

---

## API Reference

### Extract from PDF
```http
POST /api/v1/extract/pdf
Content-Type: multipart/form-data

file: <pdf_bytes>
```

### Extract from Text
```http
POST /api/v1/extract/text
Content-Type: application/json

{
  "text": "This Agreement is entered into as of January 15, 2024..."
}
```

### Response Format
```json
{
  "success": true,
  "filename": "contract.pdf",
  "entities": [
    {
      "entity_type": "DATE",
      "value": "January 15, 2024",
      "normalized_value": "2024-01-15",
      "confidence": 0.97,
      "valid": true,
      "validation_notes": "Normalized from written format"
    }
  ],
  "metadata": {
    "document_type": "PDF",
    "ocr_applied": false,
    "page_count": 12,
    "word_count": 4832,
    "processing_time_ms": 742.3,
    "confidence_score": 0.94
  }
}
```

---

## Training Your Own Model

1. Annotate contracts in [Doccano](https://github.com/doccano/doccano) with labels:
   `DATE`, `PARTY`, `AMOUNT`, `TERMINATION_CLAUSE`

2. Export as JSONL and run:
```bash
python backend/train_ner.py \
  --data ./data/annotations.jsonl \
  --output ./backend/models/lexiscan-ner \
  --iter 50
```

3. Model auto-loads from `./backend/models/lexiscan-ner` at startup.

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend Framework | FastAPI + Uvicorn |
| OCR | Tesseract + PyMuPDF |
| NER | SpaCy + BERT (en_core_web_trf) |
| Training | SpaCy training pipeline |
| Validation | Custom rule-based Python |
| Frontend | React + Vite |
| Containerization | Docker + Docker Compose |
| Cache | Redis |
| Proxy | Nginx |
