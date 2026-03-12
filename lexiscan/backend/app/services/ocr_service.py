"""
OCR Service — Tesseract + PyMuPDF Pipeline
Handles native digital PDFs, scanned PDFs, and image files.

Week 1 Implementation:
- Native PDF text extraction via PyMuPDF (no OCR needed)
- Scanned PDF handling via Tesseract OCR
- Direct image file OCR (JPG, PNG, TIFF, BMP, WEBP)
- Text quality enhancement and noise reduction
"""

import io
import re
import os
import logging
from dataclasses import dataclass
from typing import Tuple
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    ocr_applied: bool
    page_count: int
    confidence: float
    pages_text: list


class OCRService:
    """
    Intelligent text extractor — supports PDFs and image files.

    Strategy:
    1. PDF  — native text via PyMuPDF; scanned pages via Tesseract
    2. Image — Tesseract OCR directly on the image bytes
    3. Noise reduction and text normalisation applied in both cases
    """

    NATIVE_TEXT_THRESHOLD = 50

    SUPPORTED_IMAGE_FORMATS = {
        ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"
    }

    # Common Tesseract install paths on Windows
    WINDOWS_TESSERACT_PATHS = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        r"C:\Tesseract-OCR\tesseract.exe",
    ]

    def __init__(self):
        self._init_ocr_engine()

    def _init_ocr_engine(self):
        """
        Initialize Tesseract OCR.
        On Windows, auto-detects the install path if not in system PATH.
        """
        try:
            import pytesseract
            if os.name == "nt":
                self._set_windows_tesseract_path(pytesseract)
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("Tesseract OCR initialized successfully")
        except Exception:
            self.tesseract_available = False
            logger.warning(
                "Tesseract not available — OCR fallback disabled.\n"
                "  Native digital PDFs will still work fine.\n"
                "  To enable OCR for scanned PDFs and images:\n"
                "    1. Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
                "    2. Install and tick 'Add to PATH'\n"
                "    3. Restart the server"
            )

    def _set_windows_tesseract_path(self, pytesseract):
        """Check common Windows install locations and set the path if found."""
        for path in self.WINDOWS_TESSERACT_PATHS:
            if os.path.isfile(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"Tesseract found at: {path}")
                return

    # ── PDF Extraction ───────────────────────────────────────────

    async def extract_text(self, pdf_bytes: bytes) -> OCRResult:
        """Extract text from PDF bytes (native or scanned)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_extract_text, pdf_bytes
        )

    def _sync_extract_text(self, pdf_bytes: bytes) -> OCRResult:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError("PyMuPDF not installed. Run: pip install PyMuPDF==1.24.11")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = []
        ocr_applied = False
        total_confidence = 0.0

        for page_num in range(len(doc)):
            page = doc[page_num]
            native_text = page.get_text("text").strip()

            if len(native_text) >= self.NATIVE_TEXT_THRESHOLD:
                cleaned = self._clean_native_text(native_text)
                pages_text.append(cleaned)
                total_confidence += 0.98
            else:
                if self.tesseract_available:
                    ocr_text, conf = self._ocr_page(page)
                    pages_text.append(ocr_text)
                    total_confidence += conf
                    ocr_applied = True
                    logger.info(f"OCR applied to page {page_num + 1} (confidence: {conf:.2f})")
                else:
                    logger.warning(f"Page {page_num + 1} appears scanned but Tesseract is unavailable.")
                    pages_text.append(native_text)
                    total_confidence += 0.3

        doc.close()
        full_text = self._post_process_text("\n\n".join(pages_text))
        avg_confidence = total_confidence / max(len(pages_text), 1)

        return OCRResult(
            text=full_text,
            ocr_applied=ocr_applied,
            page_count=len(pages_text),
            confidence=round(avg_confidence, 3),
            pages_text=pages_text,
        )

    def _ocr_page(self, page) -> Tuple[str, float]:
        """Render a PDF page to image and run Tesseract OCR."""
        import pytesseract
        from PIL import Image

        mat = page.get_pixmap(matrix=page.fitz.Matrix(300 / 72, 300 / 72))
        img = Image.open(io.BytesIO(mat.tobytes("png")))

        config = r"--oem 3 --psm 6 -c tessedit_char_blacklist=|~^"
        data = pytesseract.image_to_data(
            img, config=config, output_type=pytesseract.Output.DICT
        )

        words, confidences = [], []
        for i, word in enumerate(data["text"]):
            conf = int(data["conf"][i])
            if conf > 30 and word.strip():
                words.append(word)
                confidences.append(conf / 100.0)

        text = " ".join(words)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return self._clean_ocr_text(text), avg_conf

    # ── Image Extraction ─────────────────────────────────────────

    async def extract_text_from_image(self, image_bytes: bytes) -> OCRResult:
        """
        Extract text directly from an image file (JPG, PNG, TIFF, BMP, WEBP).
        Tesseract OCR is applied to the full image and text is returned.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_extract_image, image_bytes
        )

    def _sync_extract_image(self, image_bytes: bytes) -> OCRResult:
        """Synchronous image OCR — runs in thread pool."""
        if not self.tesseract_available:
            raise RuntimeError(
                "Tesseract is required for image OCR but is not available. "
                "Install from https://github.com/UB-Mannheim/tesseract/wiki"
            )

        import pytesseract
        from PIL import Image

        # Open image from raw bytes
        img = Image.open(io.BytesIO(image_bytes))

        # Normalise colour mode — Tesseract works best on RGB or greyscale
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # OEM 3 = LSTM engine, PSM 6 = assume uniform block of text
        config = r"--oem 3 --psm 6"
        data = pytesseract.image_to_data(
            img, config=config, output_type=pytesseract.Output.DICT
        )

        words, confidences = [], []
        for i, word in enumerate(data["text"]):
            conf = int(data["conf"][i])
            if conf > 30 and word.strip():
                words.append(word)
                confidences.append(conf / 100.0)

        raw_text  = " ".join(words)
        clean     = self._post_process_text(self._clean_ocr_text(raw_text))
        avg_conf  = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info(f"Image OCR complete — {len(words)} words, confidence: {avg_conf:.2f}")

        return OCRResult(
            text=clean,
            ocr_applied=True,
            page_count=1,
            confidence=round(avg_conf, 3),
            pages_text=[clean],
        )

    # ── Text Cleaning ────────────────────────────────────────────

    def _clean_native_text(self, text: str) -> str:
        """Fix ligatures and normalise whitespace in native PDF text."""
        for char, replacement in {
            "\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff",
            "\ufb03": "ffi", "\ufb04": "ffl",
        }.items():
            text = text.replace(char, replacement)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _clean_ocr_text(self, text: str) -> str:
        """Remove OCR artefact lines and collapse extra whitespace."""
        lines = text.split("\n")
        cleaned = [
            l.strip() for l in lines
            if len(l.strip()) > 2 and not re.match(r"^[^a-zA-Z0-9]+$", l.strip())
        ]
        return re.sub(r"\s{2,}", " ", " ".join(cleaned)).strip()

    def _post_process_text(self, text: str) -> str:
        """Final normalisation — fix hyphens, quotes, dashes."""
        text = re.sub(r"-\n(\w)", r"\1", text)
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = re.sub(r"[\u2013\u2014]", "-", text)
        return text.strip()
