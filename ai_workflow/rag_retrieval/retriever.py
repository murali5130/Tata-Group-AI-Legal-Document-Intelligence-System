from ai_workflow.rag_retrieval.vector_store import (
    search_clauses,
    search_clauses_multi,
)


def retrieve_relevant_chunks(
    question: str,
    k: int = 5,
    only_knowledge_base: bool = False,
    document_id: str = None,
) -> list[dict]:
    """
    Returns the top-k most relevant clauses/chunks for a question.

    Each result contains:
    - text
    - clause_type
    - source
    - risk_flag
    - page_number
    """

    rows = search_clauses(
        question,
        k=k,
        only_knowledge_base=only_knowledge_base,
        document_id=document_id,
    )

    return [
        {
            "text": text,
            "clause_type": clause_type,
            "source": source_document_id,
            "risk_flag": risk_flag,
            "page_number": page_number,
        }
        for (
            text,
            clause_type,
            source_document_id,
            risk_flag,
            page_number,
        ) in rows
    ]


def retrieve_relevant_chunks_multi(
    question: str,
    document_ids: list[str],
    k: int = 5,
) -> list[dict]:
    """
    Same shape as retrieve_relevant_chunks(), but searches across
    several specific uploaded documents at once.
    """

    rows = search_clauses_multi(
        question,
        document_ids,
        k=k,
    )

    return [
        {
            "text": text,
            "clause_type": clause_type,
            "source": source_document_id,
            "risk_flag": risk_flag,
            "page_number": page_number,
        }
        for (
            text,
            clause_type,
            source_document_id,
            risk_flag,
            page_number,
        ) in rows
    ]