import os
from groq import Groq
from dotenv import load_dotenv

from ai_workflow.rag_retrieval.retriever import (
    retrieve_relevant_chunks,
    retrieve_relevant_chunks_multi,
)

from guardrails.input_guardrail import input_guardrail
from guardrails.retriever_guardrail import retrieval_guardrail
from guardrails.output_guardrail import output_guardrail

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
_MODEL = "openai/gpt-oss-120b"


def answer_question(document_id: str, question: str, k: int = 3) -> dict:
    """
    Answers a question about one specific uploaded document.

    Returns:
        {
            "answer": str,
            "blocked": bool,
            "block_reason": str | None,
            "sources": list[dict]
        }
    """

    # Layer 1: question safety / relevance
    is_safe = input_guardrail(question)

    if is_safe is None:
        return {
            "answer": "",
            "blocked": True,
            "block_reason": "Could not verify the question's safety. Please try again.",
            "sources": [],
        }

    if is_safe is False:
        return {
            "answer": "",
            "blocked": True,
            "block_reason": "This question was blocked by the security guardrail.",
            "sources": [],
        }

    # Retrieve chunks from THIS document only
    docs = retrieve_relevant_chunks(
        question,
        k=k,
        document_id=document_id,
    )

    # Layer 2: retrieval relevance
    if not retrieval_guardrail(question, docs):
        return {
            "answer": "",
            "blocked": True,
            "block_reason": "No relevant information found in this document for that question.",
            "sources": [],
        }

    # Include page number in context when available
    context_parts = []

    for d in docs:
        page = d.get("page_number")

        if page is not None:
            context_parts.append(
                f"[Page {page}]\n{d['text']}"
            )
        else:
            context_parts.append(d["text"])

    context = "\n\n".join(context_parts)

    prompt = (
        f"Answer the question using only the following context from the contract. "
        f"If the answer is not in the context, say so clearly rather than guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600,
    )

    draft_answer = response.choices[0].message.content

    # Layer 3: output safety
    if not output_guardrail(draft_answer):
        return {
            "answer": "",
            "blocked": True,
            "block_reason": (
                "Sensitive information detected in the generated answer. "
                "Response suppressed."
            ),
            "sources": docs,
        }

    return {
        "answer": draft_answer,
        "blocked": False,
        "block_reason": None,
        "sources": docs,
    }


def answer_question_multi(
    document_ids: list[str],
    question: str,
    k: int = 5,
) -> dict:
    """
    Answers a question across MULTIPLE uploaded documents.

    Example:
        "Compare the termination clauses across these contracts."

    The selected document IDs are preserved in the retrieved sources,
    along with page numbers when available.
    """

    # Layer 1: question safety / relevance
    is_safe = input_guardrail(question)

    if is_safe is None:
        return {
            "answer": "",
            "blocked": True,
            "block_reason": (
                "Could not verify the question's safety. Please try again."
            ),
            "sources": [],
        }

    if is_safe is False:
        return {
            "answer": "",
            "blocked": True,
            "block_reason": (
                "This question was blocked by the security guardrail."
            ),
            "sources": [],
        }

    # Retrieve from ALL selected documents
    docs = retrieve_relevant_chunks_multi(
        question,
        document_ids,
        k=k,
    )

    # Layer 2: retrieval relevance
    if not retrieval_guardrail(question, docs):
        return {
            "answer": "",
            "blocked": True,
            "block_reason": (
                "No relevant information found across the selected "
                "documents for that question."
            ),
            "sources": [],
        }

    # Include document name + page number in context
    context_parts = []

    for d in docs:
        source = d.get("source", "Unknown document")
        page = d.get("page_number")

        if page is not None:
            context_parts.append(
                f"[From {source} - Page {page}]\n{d['text']}"
            )
        else:
            context_parts.append(
                f"[From {source}]\n{d['text']}"
            )

    context = "\n\n".join(context_parts)

    prompt = (
        f"Answer the question using only the following context, which comes "
        f"from multiple contracts. Clearly attribute information to the correct "
        f"document when relevant. If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800,
    )

    draft_answer = response.choices[0].message.content

    # Layer 3: output safety
    if not output_guardrail(draft_answer):
        return {
            "answer": "",
            "blocked": True,
            "block_reason": (
                "Sensitive information detected in the generated answer. "
                "Response suppressed."
            ),
            "sources": docs,
        }

    return {
        "answer": draft_answer,
        "blocked": False,
        "block_reason": None,
        "sources": docs,
    }