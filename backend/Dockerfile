# ─────────────────────────────────────────────
# LexiScan Auto — Backend Dockerfile
# Week 4: Containerized Microservice
#
# Multi-stage build for lean production image
# Includes: Tesseract OCR + SpaCy + FastAPI
# ─────────────────────────────────────────────

# ── Stage 1: Build ──────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="LexiScan Auto Team"
LABEL version="1.0.0"
LABEL description="Legal Contract NER Microservice"

# Install Tesseract OCR + runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Copy HTML frontend so FastAPI can serve it at "/"
COPY ../frontend/index.html ./frontend/index.html

# Create non-root user for security
RUN useradd -m -u 1000 lexiscan && \
    chown -R lexiscan:lexiscan /app
USER lexiscan

# Environment
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
