from pathlib import Path
import os
import tempfile

from pypdf import PdfReader
from document_pipeline.ocr.ocr_engine import extract_pdf_text_with_ocr

def extract_text(file):
    filename = getattr(file, "name", None) or str(file)

    if filename.lower().endswith(".pdf"):
        # Extract text from all PDF pages, using OCR for scanned pages
        if hasattr(file, "read"):
            pdf_bytes = file.read()
            if hasattr(file, "seek"):
                file.seek(0)

            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
                temp.write(pdf_bytes)
                temp_path = temp.name

            try:
                return extract_pdf_text_with_ocr(temp_path)
            finally:
                os.unlink(temp_path)

        return extract_pdf_text_with_ocr(filename)

    # .txt or any plain-text file
    if hasattr(file, "read"):
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        return content

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
