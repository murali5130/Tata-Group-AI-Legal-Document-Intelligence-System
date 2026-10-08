from document_pipeline.parsing.text_extractor import extract_text
from ai_workflow.rag_retrieval.document_ingestion import ingest_document
from ai_workflow.legal_reasoning.chians import run_legal_analysis
from guardrails.output_guardrail import output_guardrail


def analyze_document(file) -> dict:
    """
    Analyze an uploaded PDF document while preserving verified
    PDF page numbers from the text extraction stage.

    Page numbers are taken only from extract_text() metadata.
    The language model is not responsible for generating page numbers.

    Returns:
        {
            "filename": str,
            "chunks_stored": int,
            "extracted_clauses": str,
            "risk_flags": str,
            "summary": str,
            "blocked": bool,
            "block_reason": str | None,
        }
    """

    filename = getattr(file, "name", None) or str(file)

    pages = extract_text(file)

    if not pages:
        return {
            "filename": filename,
            "chunks_stored": 0,
            "extracted_clauses": "",
            "risk_flags": "",
            "summary": "",
            "blocked": True,
            "block_reason": (
                "No readable text could be extracted from this document."
            ),
        }


    valid_pages = [
        page
        for page in pages
        if isinstance(page, dict)
        and page.get("text", "").strip()
    ]

    if not valid_pages:
        return {
            "filename": filename,
            "chunks_stored": 0,
            "extracted_clauses": "",
            "risk_flags": "",
            "summary": "",
            "blocked": True,
            "block_reason": (
                "No readable text could be extracted from this document."
            ),
        }


    chunks_stored = ingest_document(
        filename,
        valid_pages,
    )

   
    page_sections = []

    for page in valid_pages:
        page_number = page.get("page_number")
        page_text = page.get("text", "").strip()

        if not page_text:
            continue

        page_sections.append(
            f"[PDF_PAGE_NUMBER={page_number}]\n"
            f"{page_text}"
        )

    contract_text = "\n\n".join(page_sections)


    result = run_legal_analysis(contract_text)

    is_clean = output_guardrail(
        result.get("summary", "")
    )

    if not is_clean:
        return {
            "filename": filename,
            "chunks_stored": chunks_stored,
            "extracted_clauses": result.get(
                "extracted_clauses", ""
            ),
            "risk_flags": result.get(
                "risk_flags", ""
            ),
            "summary": "",
            "blocked": True,
            "block_reason": (
                "Sensitive information was detected in the generated "
                "summary. Response suppressed."
            ),
        }
        
    return {
        "filename": filename,
        "chunks_stored": chunks_stored,
        "extracted_clauses": result.get(
            "extracted_clauses", ""
        ),
        "risk_flags": result.get(
            "risk_flags", ""
        ),
        "summary": result.get(
            "summary", ""
        ),
        "blocked": False,
        "block_reason": None,
    }