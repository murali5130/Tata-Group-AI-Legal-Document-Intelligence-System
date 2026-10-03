from document_pipeline.parsing.text_extractor import extract_text
from ai_workflow.rag_retrieval.document_ingestion import ingest_document
from ai_workflow.legal_reasoning.chians import run_legal_analysis
from guardrails.output_guardrail import output_guardrail


def analyze_document(file) -> dict:
    """
    Returns:
        {
            "filename": str,
            "chunks_stored": int,     # how many chunks were saved for Q&A
            "extracted_clauses": str,
            "risk_flags": str,
            "summary": str,
            "blocked": bool,
            "block_reason": str | None,
        }
    """
    filename = getattr(file, "name", None) or str(file)
    contract_text = extract_text(file)

    if not contract_text.strip():
        return {
            "filename": filename,
            "chunks_stored": 0,
            "extracted_clauses": "",
            "risk_flags": "",
            "summary": "",
            "blocked": True,
            "block_reason": "No readable text could be extracted from this document.",
        }

    # Store this document's chunks so qa.answer_question() can find them later
    chunks_stored = ingest_document(filename, contract_text)

    result = run_legal_analysis(contract_text)

    is_clean = output_guardrail(result["summary"])
    if not is_clean:
        return {
            "filename": filename,
            "chunks_stored": chunks_stored,
            "extracted_clauses": result["extracted_clauses"],
            "risk_flags": result["risk_flags"],
            "summary": "",
            "blocked": True,
            "block_reason": (
                "Sensitive information was detected in the generated summary. "
                "Response suppressed."
            ),
        }

    return {
        "filename": filename,
        "chunks_stored": chunks_stored,
        "extracted_clauses": result["extracted_clauses"],
        "risk_flags": result["risk_flags"],
        "summary": result["summary"],
        "blocked": False,
        "block_reason": None,
    }
