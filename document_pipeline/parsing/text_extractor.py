from pathlib import Path
import os
import tempfile

from pypdf import PdfReader
from document_pipeline.ocr.ocr_engine import extract_pdf_text_with_ocr


def extract_text(file):
    filename = getattr(file, "name", None) or str(file)

    if filename.lower().endswith(".pdf"):
        # Extract text page-by-page.
        # OCR is still used for scanned pages.
        if hasattr(file, "read"):
            pdf_bytes = file.read()

            if hasattr(file, "seek"):
                file.seek(0)

            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False
            ) as temp:
                temp.write(pdf_bytes)
                temp_path = temp.name

            try:
                return _extract_pdf_pages_with_metadata(temp_path)
            finally:
                os.unlink(temp_path)

        return _extract_pdf_pages_with_metadata(filename)

    # .txt or any plain-text file
    if hasattr(file, "read"):
        content = file.read()

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        return [{
            "page_number": None,
            "text": content
        }]

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        return [{
            "page_number": None,
            "text": f.read()
        }]


def _extract_pdf_pages_with_metadata(pdf_path):
    """
    Extract PDF text page-by-page while preserving page numbers.

    Normal PDFs:
        pypdf extracts the page text directly.

    Scanned PDFs:
        OCR is used for pages where normal extraction produces
        little or no useful text.

    Returns:
        [
            {
                "page_number": 1,
                "text": "..."
            },
            {
                "page_number": 2,
                "text": "..."
            }
        ]
    """

    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        # First try normal PDF text extraction
        page_text = page.extract_text() or ""

        # If the page has usable text, keep it
        if page_text.strip():
            pages.append({
                "page_number": page_number,
                "text": page_text
            })
            continue

        # Otherwise use OCR for this page/document
        try:
            ocr_result = extract_pdf_text_with_ocr(pdf_path)

            # If the existing OCR function returns page-wise data
            if isinstance(ocr_result, list):

                for item in ocr_result:
                    if isinstance(item, dict):
                        ocr_page_number = item.get("page_number")
                        ocr_text = item.get("text", "")

                        if ocr_page_number == page_number:
                            pages.append({
                                "page_number": page_number,
                                "text": ocr_text
                            })
                            break

                else:
                    pages.append({
                        "page_number": page_number,
                        "text": ""
                    })

            # If OCR returns one large string, we cannot safely
            # determine individual page boundaries here.
            else:
                pages.append({
                    "page_number": page_number,
                    "text": str(ocr_result)
                })

        except Exception:
            pages.append({
                "page_number": page_number,
                "text": ""
            })

    return pages