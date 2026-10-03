import os
from groq import Groq
from dotenv import load_dotenv

from ai_workflow.rag_retrieval.retriever import retrieve_relevant_chunks
from data.clause_list import clause_list

from guardrails.input_guardrail import input_guardrail
from guardrails.retriever_guardrail import retrieval_guardrail

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
_MODEL = "openai/gpt-oss-120b"

TARGET_CLAUSE_TYPES = clause_list
def _ask(prompt: str, max_tokens: int = 1500) -> str:
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def extract_clauses(contract_text: str) -> str:
    clause_types_str = ", ".join(TARGET_CLAUSE_TYPES)

    # Split the contract into smaller sections to reduce request size.
    chunk_size = 12000
    text_chunks = [
        contract_text[i:i + chunk_size]
        for i in range(0, len(contract_text), chunk_size)
    ]

    all_results = []

    for index, text_chunk in enumerate(text_chunks, start=1):
        prompt = (
            "You are a legal document clause extraction assistant.\n\n"
            f"Extract all relevant clauses belonging to these categories, "
            f"if present: {clause_types_str}.\n\n"

            "IMPORTANT INSTRUCTIONS:\n"
            "1. Examine the entire provided section carefully.\n"
            "2. Extract EVERY relevant clause, not just the first match.\n"
            "3. Preserve the exact wording of the source text as far as "
            "it is available. Do not invent missing text.\n"
            "4. A clause may relate to a category even if its heading "
            "does not use the category name. Identify it by its meaning.\n"
            "5. Include the clause number or section number when available.\n"
            "6. Do not classify a clause as absent from the entire document "
            "because it is missing from this section.\n"
            "7. Do not generate risk ratings or legal conclusions here.\n"
            "8. If no relevant clause is found in this section, return "
            "'No matching clauses found in this section.'\n\n"

            "For every clause found, use this format:\n"
            "✅ [CLAUSE CATEGORY] | Exact extracted text: [verbatim text]\n\n"

            f"DOCUMENT SECTION {index} OF {len(text_chunks)}:\n"
            f"{text_chunk}\n"
        )

        result = _ask(prompt, max_tokens=1500)
        all_results.append(result)

    return "\n\n".join(all_results)

def classify_risk(
    extracted_clauses: str,
    document_type: str = "Unknown"
) -> str:
    """
    Ground risk analysis against approved Tata Group knowledge-base
    standards using clause-focused retrieval.
    Only clauses actually identified as present are used for retrieval.
    Apply policy standards according to the document type.
    """

    present_lines = []

    for line in extracted_clauses.splitlines():
        stripped = line.strip()

        # Ignore categories explicitly marked as not present
        if "❌" in stripped or "Not present" in stripped:
            continue

        # Keep lines that contain extracted clause text
        if "✅" in stripped or "| *" in stripped:
            present_lines.append(stripped)

    if not present_lines:
        return (
            "No substantive contract clauses were identified for "
            "risk analysis."
        )

    # Keep the retrieval query focused
    retrieval_query = "\n".join(present_lines)

    reference_hits = retrieve_relevant_chunks(
        retrieval_query,
        k=10,
        only_knowledge_base=True
    )

    retrieval_ok = retrieval_guardrail(
        retrieval_query,
        reference_hits
    )

    if not retrieval_ok:
        return (
            "Risk analysis stopped because no sufficiently relevant "
            "approved Tata Group policy reference was found."
        )

    # Remove duplicate policy references
    unique_hits = []
    seen = set()

    for hit in reference_hits:
        key = (
            hit.get("clause_type", ""),
            hit.get("text", "")
        )

        if key not in seen:
            seen.add(key)
            unique_hits.append(hit)

    reference_text = "\n".join(
        f"- [{r['clause_type']}] {r['text']}"
        for r in unique_hits
    )

    prompt = (
        "You are analyzing a document against approved Tata Group "
        "policy standards.\n\n"

        f"DOCUMENT TYPE: {document_type}\n\n"

        "IMPORTANT RULES:\n"
        "1. Use ONLY the extracted clauses and approved policy "
        "standards provided below.\n"
        "2. Do not use outside legal knowledge.\n"
        "3. Do not invent missing contract language.\n"
        "4. Do not assume that an indemnity is outside a liability "
        "cap unless the contract text or policy reference explicitly "
        "supports that conclusion.\n"
        "5. Do not treat a missing clause as a risk unless the "
        "approved policy reference establishes that it is required.\n"
        "6. If there is insufficient evidence to determine a risk, "
        "state 'Insufficient evidence' instead of guessing.\n"
        "7. Apply policy standards only when relevant to the "
        "identified document type.\n"
        "8. If the document type is MOA/AOA, do NOT treat the "
        "members' limited liability statement as a vendor-contract "
        "limitation-of-liability cap.\n"
        "9. If the document is MOA/AOA, do not flag missing "
        "vendor-contract requirements unless an approved policy "
        "explicitly establishes that the requirement applies to "
        "MOA/AOA documents.\n"
        "10. If a retrieved policy is specific to vendor contracts "
        "and the document is MOA/AOA, state that the policy is "
        "not directly applicable instead of reporting a deviation.\n"
        "11. Do not interpret an irrelevant policy match as evidence "
        "of a risk.\n\n"

        "Extracted clauses actually present in the document:\n"
        f"{retrieval_query}\n\n"

        "Approved Tata Group policy standards retrieved from the "
        "knowledge base:\n"
        f"{reference_text}\n\n"

        "Compare each extracted clause only with relevant approved "
        "policy standards.\n\n"

        "Return risk findings with:\n"
        "- clause_type\n"
        "- severity (low/medium/high)\n"
        "- finding\n"
        "- rationale\n"
        "- policy_reference\n\n"

        "If no applicable policy deviation is supported by the "
        "evidence, explicitly state that no applicable policy "
        "deviation was identified.\n"
    )

    return _ask(prompt)


def summarize(extracted_clauses: str,
              risk_flags: str,
              document_type: str = "Unknown"
            ):
    prompt = (
        "You are preparing a factual legal-document review summary "
        "for a Tata Group legal team member.\n\n"
        f"Document type identified by the system: {document_type}\n\n"
        "Use this document type in the summary. Do not independently "
        "guess a different document type. If the type is Unknown, "
        "state that the document type could not be confidently identified.\n\n"

        "IMPORTANT INSTRUCTIONS:\n"
        "1. Identify the document type from the extracted content. "
        "For example, Memorandum and Articles of Association (MOA/AOA), "
        "vendor agreement, NDA, or service agreement. If unclear, "
        "state 'Document type unclear'.\n"
        "2. Explain the document's actual purpose and the important "
        "clauses found in the extracted text.\n"
        "3. Preserve the meaning and context of each clause.\n"
        "4. Do not treat a company's or its members' limited liability "
        "provision as a contractual liability cap unless the text "
        "explicitly supports that interpretation.\n"
        "5. Do not describe a clause as missing unless the available "
        "document text is sufficient to establish its absence.\n"
        "6. Do not invent clauses, obligations, risks, or legal "
        "conclusions. Use only the extracted clauses and risk findings.\n"
        "7. If evidence is insufficient, clearly say so.\n"
        "8. Keep the summary under 200 words.\n\n"
        "9. Use the document type identified by the system above. "
        "Explain the document's actual purpose only when supported "
        "by the extracted text. Do not guess the document type.\n"

        "Use this format:\n"
        "Document Type:\n"
        "Document Purpose:\n"
        "Key Clauses:\n"
        "Risk Findings:\n"
        "Limitations of Review:\n\n"

        f"Extracted clauses:\n{extracted_clauses}\n\n"
        f"Risk findings:\n{risk_flags}\n\n"

        "Write a concise, factual summary. Distinguish actual "
        "document provisions from potential risks. Do not give "
        "unsupported recommendations."
    )

    return _ask(prompt, max_tokens=500)

def run_legal_analysis(contract_text: str) -> dict:
    """
    Runs input guardrail -> extraction -> document type detection
    -> retrieval guardrail -> risk classification -> summary.
    """

    input_status = input_guardrail(contract_text)

    if input_status is False:
        return {
            "extracted_clauses": "",
            "risk_flags": "",
            "summary": (
                "Request blocked by the input guardrail. "
                "The provided input was identified as unsafe."
            ),
        }

    if input_status is None:
        return {
            "extracted_clauses": "",
            "risk_flags": "",
            "summary": (
                "Input validation could not be completed. "
                "Please try again."
            ),
        }

    extracted_clauses = extract_clauses(contract_text)

    text_lower = contract_text.lower()

    if (
        "memorandum and articles of association" in text_lower
        or (
            "memorandum of association" in text_lower
            and "articles of association" in text_lower
        )
        or (
            "memorandum" in text_lower
            and "articles of association" in text_lower
        )
        or (
            "revised memorandum" in text_lower
            and "articles of association" in text_lower
        )
    ):
        document_type = "MOA/AOA"

    elif (
        "non-disclosure agreement" in text_lower
        or "confidentiality agreement" in text_lower
    ):
        document_type = "NDA"

    elif "service agreement" in text_lower:
        document_type = "Service Agreement"

    elif (
        "vendor agreement" in text_lower
        or "vendor contract" in text_lower
    ):
        document_type = "Vendor Agreement"

    else:
        document_type = "Unknown"

    risk_flags = classify_risk(
        extracted_clauses,
        document_type
    )

    summary = summarize(
        extracted_clauses,
        risk_flags,
        document_type
    )

    return {
        "extracted_clauses": extracted_clauses,
        "risk_flags": risk_flags,
        "summary": summary,
    }
