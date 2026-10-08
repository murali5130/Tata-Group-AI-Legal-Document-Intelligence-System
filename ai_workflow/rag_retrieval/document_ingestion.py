from ai_workflow.rag_retrieval.chunking import chunk_text
from ai_workflow.rag_retrieval.vector_store import store_clause, delete_document


def ingest_document(
    filename: str,
    contract_text: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200
) -> int:
    """
    Chunks and stores an uploaded contract's page-wise text in pgvector.

    Each chunk keeps its original PDF page number so that
    source citations can be displayed later.

    Returns the number of chunks stored.
    """

    chunks = chunk_text(
        contract_text,
        chunk_size=chunk_size,
        overlap=overlap
    )

    # Re-uploading the same filename replaces old chunks
    # instead of duplicating them.
    delete_document(filename)

    for chunk in chunks:
        store_clause(
            document_id=filename,
            clause_type="UPLOADED_CONTRACT",
            clause_text=chunk["text"],
            risk_flag=None,
            page_number=chunk["page_number"],
        )

    return len(chunks)