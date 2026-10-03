
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader


# Windows lo Tesseract executable location
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Poppler executable folder
POPPLER_PATH = (
    r"C:\Users\HP\OneDrive\Documents"
    r"\poppler-26.09.0\Library\bin"
)


def extract_pdf_text_with_ocr(pdf_path: str) -> str:
    """
    Extract text from every PDF page.
    Use OCR only when a page has little or no extractable text.
    """
    reader = PdfReader(pdf_path)
    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        # Scanned pages usually have little or no selectable text
        if len(text) < 30:
            print(f"Page {page_number}: Running OCR...")

            images = convert_from_path(
                pdf_path,
                first_page=page_number,
                last_page=page_number,
                dpi=200,
                poppler_path=POPPLER_PATH,
            )

            if images:
                text = pytesseract.image_to_string(
                    images[0],
                    lang="eng",
                ).strip()

        extracted_pages.append(
            f"\n--- PAGE {page_number} ---\n{text}"
        )

    return "\n".join(extracted_pages)
