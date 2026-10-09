
import os
import shutil
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader


# Configure Tesseract only when a Windows executable is available.
tesseract_path = shutil.which("tesseract")

if os.name == "nt":
    windows_tesseract = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    if Path(windows_tesseract).exists():
        tesseract_path = windows_tesseract

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


def extract_pdf_text_with_ocr(pdf_path: str) -> str:
    """
    Extract text page by page.
    Run OCR only when a page has little or no extractable text.
    Supports Windows and Linux deployments.
    """
    reader = PdfReader(pdf_path)
    extracted_pages = []

    poppler_path = None

    # On Windows, use the configured Poppler path if it exists.
    if os.name == "nt":
        windows_poppler = (
            r"C:\Users\HP\OneDrive\Documents"
            r"\poppler-26.09.0\Library\bin"
        )
        if Path(windows_poppler).exists():
            poppler_path = windows_poppler

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if len(text) < 30:
            print(f"Page {page_number}: Running OCR...")

            try:
                images = convert_from_path(
                    pdf_path,
                    first_page=page_number,
                    last_page=page_number,
                    dpi=200,
                    poppler_path=poppler_path,
                )

                if images:
                    text = pytesseract.image_to_string(
                        images[0],
                        lang="eng",
                    ).strip()

                if not text:
                    print(
                        f"Page {page_number}: OCR returned no text."
                    )

            except Exception as exc:
                print(
                    f"Page {page_number}: OCR failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        extracted_pages.append(
            f"\n--- PAGE {page_number} ---\n{text}"
        )

    return "\n".join(extracted_pages)
