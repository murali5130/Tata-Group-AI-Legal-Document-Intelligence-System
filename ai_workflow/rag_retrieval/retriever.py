from ai_workflow.rag_retrieval.vector_store import search_clauses


def retrieve_relevant_chunks(
    question: str,
    k: int = 5,
    only_knowledge_base: bool = False,
    document_id: str = None,
) -> list[dict]:
    """
    Returns the top-k most relevant clauses/chunks for a question, as
    plain dicts: [{"text": ..., "clause_type": ..., "source": ..., "risk_flag": ...}, ...]

    Pass document_id to search within one specific uploaded document only
    (the Q&A use case). Pass only_knowledge_base=True to search the
    approved reference clauses only (the risk-grounding use case).
    """
    rows = search_clauses(
        question, k=k, only_knowledge_base=only_knowledge_base, document_id=document_id
    )
    return [
        {"text": text, "clause_type": clause_type, "source": document_id, "risk_flag": risk_flag}
        for text, clause_type, document_id, risk_flag in rows
    ]
