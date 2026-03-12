# LexiScan Auto — Project Structure & Run Guide

---

## Folder Structure

```
lexiscan/                              ← Root project folder
│
├── docker-compose.yml                 ← Runs the whole stack with one command
│
├── data/                              ← Input files
│   ├── lexiscan_test_contract.pdf
│   ├── sample_contract_page1.png
│   ├── sample_contract_page2.png
│   ├── sample_contract_page3.png
│   ├── sample_contract_page4.png
│   └── sample_contract_page5.png             
│
├── frontend/
│   └── index.html                     ← Entire UI (HTML + CSS + JS, single file)
│
└── backend/
    ├── Dockerfile                     ← Builds the backend container
    ├── requirements.txt               ← All Python dependencies
    ├── train_ner.py                   ← Script to train/fine-tune the NER model
    │
    └── app/
        ├── main.py                    ← FastAPI app entry point; also serves index.html
        │
        ├── api/
        │   └── routes.py              ← API endpoints: /extract/pdf, /extract/text, /model/info
        │
        ├── models/
        │   └── schemas.py             ← Pydantic request/response data models
        │
        └── services/
            ├── ocr_service.py         ← Week 1: Tesseract OCR + PyMuPDF text extraction
            ├── ner_service.py         ← Week 1-2: SpaCy NER + regex entity extraction
            └── validation_service.py  ← Week 3: Rule-based normalization & validation
```

---

## What Each File Does

| File | Purpose |
|------|---------|
| `index.html` | The complete frontend — drag-drop PDF upload, paste text, displays entity results |
| `main.py` | Starts the FastAPI server; serves `index.html` at `/`; mounts API at `/api/v1` |
| `routes.py` | Defines `POST /extract/pdf` and `POST /extract/text` endpoints |
| `schemas.py` | Pydantic models that define the exact JSON shape of requests and responses |
| `ocr_service.py` | Detects if PDF is scanned → runs Tesseract OCR; or extracts native text via PyMuPDF |
| `ner_service.py` | Runs SpaCy NER model + regex patterns to find DATE, PARTY, AMOUNT, TERMINATION_CLAUSE |
| `validation_service.py` | Normalizes dates to YYYY-MM-DD, amounts to $X,XXX.XX, validates party names |
| `train_ner.py` | Standalone script to fine-tune SpaCy on your own annotated legal contracts |
| `requirements.txt` | Lists every Python package needed |
| `Dockerfile` | Packages the backend + Tesseract + SpaCy into a Docker container |
| `docker-compose.yml` | Orchestrates the API container + Redis cache with one command |

---

## How to Run — Option A: Locally (Recommended for Development)

### Step 1 — Prerequisites

Make sure you have these installed:

```
Python 3.10 or higher    →  https://python.org
pip                      →  comes with Python
Tesseract OCR            →  see below
```

**Install Tesseract:**

- **Windows:** Download installer from https://github.com/UB-Mannheim/tesseract/wiki
  After install, add to PATH: `C:\Program Files\Tesseract-OCR`
- **macOS:** `brew install tesseract`
- **Linux/Ubuntu:** `sudo apt-get install tesseract-ocr`

---

### Step 2 — Clone / Download the Project

```bash
# If using git:
git clone <your-repo-url>
cd lexiscan

# Or just unzip the downloaded folder and open a terminal inside it
cd lexiscan
```

---

### Step 3 — Create a Python Virtual Environment

```bash
# Inside the lexiscan/ folder:
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

---

### Step 4 — Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This installs FastAPI, SpaCy, PyMuPDF, Tesseract wrapper, Pydantic, and all other packages.
It also downloads the SpaCy `en_core_web_sm` language model automatically.

> **Note:** First install takes 2–5 minutes. SpaCy model download is included.

---

### Step 5 — Start the Backend Server

```bash
# Still inside the backend/ folder:
uvicorn app.main:app --reload --port 8000
```

You will see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

---

### Step 6 — Open the App

Open your browser and go to:

```
http://localhost:8000
```

The frontend (`index.html`) is served automatically by FastAPI.

Also available:
```
http://localhost:8000/docs        ← Interactive Swagger API docs
http://localhost:8000/health      ← Health check endpoint
```

---

### Step 7 — Try It

**Option A — Paste Text:**
1. The text tab is pre-filled with a sample contract
2. Click **Extract Entities**
3. Results appear on the right with dates, parties, amounts, termination clauses

**Option B — Upload a PDF:**
1. Click the "PDF Upload" tab
2. Drag and drop any PDF contract, or click to browse
3. Click **Extract Entities**
4. The pipeline runs: OCR (if scanned) → NER → Validation → Results

---

## How to Run — Option B: Docker (Recommended for Production)

### Prerequisites

- Docker Desktop installed: https://www.docker.com/products/docker-desktop

### Step 1 — Build and Start

```bash
# From the root lexiscan/ folder:
docker-compose up --build
```

First build takes 5–10 minutes (downloads Tesseract, SpaCy, all dependencies).

### Step 2 — Open the App

```
http://localhost:8000
```

### Stop the app

```bash
docker-compose down
```

---

## Request & Response Flow

```
User uploads PDF or pastes text
        ↓
  POST /api/v1/extract/pdf   (or /extract/text)
        ↓
  [ocr_service.py]
   - Native PDF? → PyMuPDF extracts text directly
   - Scanned PDF? → Tesseract OCR converts image → text
        ↓
  [ner_service.py]
   - SpaCy NER model identifies entity spans
   - Regex patterns supplement for DATE/AMOUNT
   - Overlapping entities are deduplicated
        ↓
  [validation_service.py]
   - DATE  → parsed & normalized to YYYY-MM-DD
   - AMOUNT → normalized to $X,XXX.XX
   - PARTY → cleaned of noise, abbreviations standardized
   - TERMINATION_CLAUSE → keyword validation, length check
        ↓
  JSON response returned to browser
        ↓
  Frontend renders entity cards with confidence bars
```

---

## API Endpoints Quick Reference

| Method | URL | What it does |
|--------|-----|-------------|
| `GET` | `/` | Serves the HTML frontend |
| `GET` | `/health` | Returns server health status |
| `POST` | `/api/v1/extract/pdf` | Upload a PDF, returns extracted entities as JSON |
| `POST` | `/api/v1/extract/text` | Post raw text, returns extracted entities as JSON |
| `GET` | `/api/v1/model/info` | Returns NER model metadata and F1 scores |
| `GET` | `/api/v1/stats` | Returns processing statistics |
| `GET` | `/docs` | Swagger interactive API documentation |

---

## Training a Custom NER Model (Optional — Week 2)

If you have annotated contracts from Doccano:

```bash
# From backend/ folder:
python train_ner.py \
  --data ./data/annotations.jsonl \
  --output ./models/lexiscan-ner \
  --iter 50
```

Once trained, the model saves to `./models/lexiscan-ner/` and is automatically loaded
on the next server start instead of the default SpaCy model.

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `TesseractNotFoundError` | Tesseract is not installed or not in PATH. See Step 1 |
| `ModuleNotFoundError: spacy` | Run `pip install -r requirements.txt` again |
| `Port 8000 already in use` | Kill the other process or use `--port 8001` |
| `CORS error in browser` | Backend not running; check terminal for errors |
| `SpaCy model not found` | Run `python -m spacy download en_core_web_sm` manually |
