from guardrails.input_guardrail import _ask_guardrail


def retrieval_guardrail(user_query: str, docs: list) -> bool:
    """
    Checks whether retrieved knowledge-base documents are relevant
    to the user's query / extracted clauses.

    Returns:
        True  -> sufficient relevant context was retrieved
        False -> no sufficiently relevant context was retrieved
    """
    if not docs:
        return False
    # Extract only usable text from retrieved results
    retrieved_texts = [
        doc.get("text", "").strip()
        for doc in docs
        if isinstance(doc, dict) and doc.get("text", "").strip()
    ]
    if not retrieved_texts:
        return False
    context = "\n\n".join(retrieved_texts)
    prompt = f"""
                You are a retrieval relevance guardrail for a Tata Group
                legal document analysis system.
                Your ONLY task is to determine whether the retrieved approved
                knowledge-base context is relevant to the query.
                Do NOT perform legal analysis.
                Do NOT determine risk.
                Do NOT give legal advice.
                Do NOT judge whether the contract complies with policy.
Query:
{user_query}
Retrieved approved knowledge-base context:
{context}
Rules:
1. Reply RELEVANT if the retrieved context contains information
   that can reasonably help analyze the query.
2. Partial semantic matches ARE relevant.
   Exact wording is NOT required.
3. Related legal concepts, clause categories, requirements,
   standards, exceptions, limitations, or obligations can count
   as relevant.
4. Reply IRRELEVANT only when the retrieved context is clearly
   unrelated to the query.
5. If at least one retrieved passage is meaningfully related,
   reply RELEVANT.
Reply with ONLY one word:
RELEVANT
or
IRRELEVANT
"""
    try:
        raw_text = _ask_guardrail(prompt)
        if not raw_text:
            return False
        decision = raw_text.strip().upper()
        if "RELEVANT" in decision:
            return True

        if "IRRELEVANT" in decision:
            return False
        return False
    except Exception:
        return False
