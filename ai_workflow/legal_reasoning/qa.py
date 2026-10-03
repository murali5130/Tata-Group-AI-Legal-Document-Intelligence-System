import os
from groq import Groq
from dotenv import load_dotenv

from ai_workflow.rag_retrieval.retriever import retrieve_relevant_chunks
from guardrails.input_guardrail import input_guardrail
from guardrails.retriever_guardrail import retrieval_guardrail
from guardrails.output_guardrail import output_guardrail

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
_MODEL = "openai/gpt-oss-120b"


def answer_question(document_id: str, question: str, k: int = 3) -> dict:
    """
    Answers a question about one specific uploaded document.

    document_id must match the filename used when the document was
    ingested via ai_workflow.rag_retrieval.document_ingestion.ingest_document().

    Returns:
        {
            "answer": str,
            "blocked": bool,
            "block_reason": str | None,
            "sources": list[dict],   # the chunks actually used, for transparency
        }
    """
    # Layer 1: is the question itself safe / on-topic?
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

    # Retrieve chunks from THIS document only, not the whole knowledge base
    docs = retrieve_relevant_chunks(question, k=k, document_id=document_id)

    # Layer 2: is what we retrieved actually relevant to the question?
    if not retrieval_guardrail(question, docs):
        return {
            "answer": "",
            "blocked": True,
            "block_reason": "No relevant information found in this document for that question.",
            "sources": [],
        }

    context = "\n\n".join(d["text"] for d in docs)
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

    # Layer 3: does the answer leak anything sensitive?
    if not output_guardrail(draft_answer):
        return {
            "answer": "",
            "blocked": True,
            "block_reason": "Sensitive information detected in the generated answer. Response suppressed.",
            "sources": docs,
        }

    return {
        "answer": draft_answer,
        "blocked": False,
        "block_reason": None,
        "sources": docs,
    }
