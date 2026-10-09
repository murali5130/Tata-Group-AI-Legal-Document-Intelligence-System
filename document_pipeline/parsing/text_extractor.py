
import os
import tempfile

from pypdf import PdfReader
from document_pipeline.ocr.ocr_engine import (
    extract_pdf_text_with_ocr,
)


def extract_text(file):
    filename = getattr(file, "name", None) or str(file)

    if filename.lower().endswith(".pdf"):
        if hasattr(file, "read"):
            pdf_bytes = file.read()

            if hasattr(file, "seek"):
                file.seek(0)

            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False,
            ) as temp:
                temp.write(pdf_bytes)
                temp_path = temp.name

            try:
                return _extract_pdf_pages_with_metadata(temp_path)
            finally:
                os.unlink(temp_path)

        return _extract_pdf_pages_with_metadata(filename)

    # .txt or other plain-text files
    if hasattr(file, "read"):
        content = file.read()

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        return [{
            "page_number": None,
            "text": content,
        }]

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        return [{
            "page_number": None,
            "text": f.read(),
        }]


def _extract_pdf_pages_with_metadata(pdf_path):
    """
    Extract text page by page and preserve page numbers.
    Use OCR for pages with little or no selectable text.
    """
    reader = PdfReader(pdf_path)
    pages = []
    ocr_pages = set()

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()

        if len(page_text) >= 30:
            pages.append({
                "page_number": page_number,
                "text": page_text,
            })
        else:
            # Mark pages that require OCR.
            pages.append({
                "page_number": page_number,
                "text": "",
            })
            ocr_pages.add(page_number)

    if ocr_pages:
        # The OCR engine processes the PDF once. It handles
        # Windows/Linux executable paths in ocr_engine.py.
        ocr_result = extract_pdf_text_with_ocr(pdf_path)

        # The OCR engine returns a string containing page markers.
        # Split that output to recover individual page text.
        current_page = None
        page_texts = {}

        for line in ocr_result.splitlines():
            if line.startswith("--- PAGE ") and line.endswith(" ---"):
                try:
                    current_page = int(
                        line.removeprefix("--- PAGE ").removesuffix(" ---")
                    )
                    page_texts[current_page] = []
                except ValueError:
                    current_page = None
            elif current_page is not None:
                page_texts[current_page].append(line)

        for item in pages:
            page_number = item["page_number"]
            if page_number in ocr_pages:
                item["text"] = "\n".join(
                    page_texts.get(page_number, [])
                ).strip()

    return pages