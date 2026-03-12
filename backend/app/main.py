"""
LexiScan Auto - Legal Contract Entity Extractor
FastAPI Backend Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.api.routes import router
import uvicorn

# Resolve the frontend directory (sits at ../../frontend relative to this file)
BASE_DIR    = Path(__file__).resolve().parent.parent          # /app
STATIC_DIR  = BASE_DIR.parent / "frontend"                   # /frontend

app = FastAPI(
    title="LexiScan Auto API",
    description="Intelligent Legal Document Processing & Named Entity Recognition",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes (/api/v1/...) ─────────────────
app.include_router(router, prefix="/api/v1")


# ── Health check ─────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "LexiScan Auto", "version": "1.0.0"}


# ── Serve frontend HTML at root ───────────────
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the HTML frontend from /frontend/index.html."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {"error": "Frontend not found. Place index.html in the /frontend directory."}


# ── Serve any other static assets (CSS, images, etc.) ───
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
