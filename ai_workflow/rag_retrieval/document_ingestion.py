from ai_workflow.rag_retrieval.chunking import chunk_text
from ai_workflow.rag_retrieval.vector_store import store_clause, delete_document


def ingest_document(filename: str, contract_text: str, chunk_size: int = 1000, overlap: int = 200) -> int:
    """
    Chunks and stores an uploaded contract's text in pgvector.
    Returns the number of chunks stored.

    Call this once per uploaded document, before asking questions about it.
    """
    chunks = chunk_text(contract_text, chunk_size=chunk_size, overlap=overlap)

    # Re-uploading the same filename replaces old chunks instead of duplicating them
    delete_document(filename)

    for chunk in chunks:
        store_clause(
            document_id=filename,          # real filename — NOT prefixed with KNOWLEDGE_BASE::
            clause_type="UPLOADED_CONTRACT",
            clause_text=chunk,
            risk_flag=None,
        )

    return len(chunks)
