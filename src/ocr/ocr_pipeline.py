"""
Week 1 - OCR Pipeline
Handles both native digital and scanned PDF documents.
Uses PyMuPDF for native text extraction and Tesseract for scanned pages.
"""

import io
import logging
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)


class OCRPipeline:
    """
    Unified OCR pipeline that:
    1. Tries native PDF text extraction (fast, clean)
    2. Falls back to Tesseract OCR for scanned/image-based pages
    3. Applies noise reduction & text normalization
    """

    NATIVE_TEXT_MIN_CHARS = 50  # threshold to consider a page "digital"

    def __init__(self, tesseract_cmd: Optional[str] = None, dpi: int = 300):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.dpi = dpi
        self.tesseract_config = r"--oem 3 --psm 6"  # LSTM OCR + assume uniform block

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract full text from a PDF file (digital or scanned)."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(path))
        pages_text = []
        for page_num, page in enumerate(doc):
            text = self._extract_page(page, page_num)
            pages_text.append(text)
        doc.close()

        full_text = "\n\n".join(pages_text)
        return self._normalize_text(full_text)

    def extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from raw PDF bytes (for API uploads)."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = []
        for page_num, page in enumerate(doc):
            text = self._extract_page(page, page_num)
            pages_text.append(text)
        doc.close()

        full_text = "\n\n".join(pages_text)
        return self._normalize_text(full_text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_page(self, page: fitz.Page, page_num: int) -> str:
        """Try native extraction; fall back to Tesseract if page is scanned."""
        native_text = page.get_text("text").strip()
        if len(native_text) >= self.NATIVE_TEXT_MIN_CHARS:
            logger.debug("Page %d: native text (%d chars)", page_num, len(native_text))
            return native_text

        logger.info("Page %d: scanned — running Tesseract OCR", page_num)
        return self._ocr_page(page)

    def _ocr_page(self, page: fitz.Page) -> str:
        """Render a PDF page to an image and run Tesseract OCR."""
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        image = self._preprocess_image(image)
        text = pytesseract.image_to_string(image, config=self.tesseract_config)
        return text.strip()

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR accuracy."""
        image = image.convert("L")
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        image = image.filter(ImageFilter.SHARPEN)
        image = image.point(lambda x: 0 if x < 140 else 255, "1")
        return image.convert("RGB")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Clean OCR artifacts and normalize whitespace."""
        # Remove non-printable control chars
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Fix common OCR ligature artifacts
        for old, new in {"\ufb01": "fi", "\ufb02": "fl", "\u2014": "--",
                         "\u2013": "-", "\u201c": '"', "\u201d": '"',
                         "\u2018": "'", "\u2019": "'"}.items():
            text = text.replace(old, new)
        return text.strip()

if __name__ == "__main__":
    # Initialize the OCR pipeline
    print("Initializing OCR Pipeline...")
    ocr = OCRPipeline()
    
    # Define a path to a test PDF (replace with your actual PDF path)
    test_pdf_path = r"C:\Users\HP\Downloads\New folder\LexiScan_Auto_Dataset_removed.pdf"  # Update this to a real PDF you have
    
    try:
        print(f"Extracting text from: {test_pdf_path}")
        extracted_text = ocr.extract_text_from_pdf(test_pdf_path)
        
        print("\n--- EXTRACTED TEXT ---")
        print(extracted_text)
        print("----------------------")
    except Exception as e:
        print(f"Error: {e}")