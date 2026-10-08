import os
import time
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
def _ask(prompt: str, max_tokens: int = 1000) -> str:
    """
    Call Groq with a small retry for temporary rate-limit errors.
    """

    for attempt in range(3):
        try:
            response = _client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            error_message = str(e)

            # Retry only for temporary rate-limit errors.
            if "429" in error_message or "rate_limit_exceeded" in error_message:
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue

            raise
def extract_clauses(contract_text: str) -> str:
    import re

    clause_types_str = ", ".join(TARGET_CLAUSE_TYPES)

    page_pattern = re.compile(
        r"\[PDF_PAGE_NUMBER=(\d+)\]\s*(.*?)(?="
        r"\[PDF_PAGE_NUMBER=\d+\]|\Z)",
        re.DOTALL,
    )

    page_sections = page_pattern.findall(contract_text)

    # Fallback for safety if page markers are not present.
    if not page_sections:
        page_sections = [("unknown", contract_text)]

    all_results = []


    for page_number, page_text in page_sections:

        page_text = page_text.strip()

        if not page_text:
            continue

        # Keep request-size protection for very large pages.
        chunk_size = 12000

        text_chunks = [
            page_text[i:i + chunk_size]
            for i in range(0, len(page_text), chunk_size)
        ]

        for chunk_index, text_chunk in enumerate(
            text_chunks,
            start=1,
        ):

            prompt = (
                "You are a legal document clause extraction assistant.\n\n"

                f"Extract all relevant clauses belonging to these "
                f"categories, if present: {clause_types_str}.\n\n"

                "IMPORTANT INSTRUCTIONS:\n"
                "1. Examine the entire provided page section carefully.\n"
                "2. Extract EVERY relevant clause found in this page section.\n"
                "3. Preserve the exact wording of the source text as far "
                "as it is available. Do not invent missing text.\n"
                "4. A clause may relate to a category even if its heading "
                "does not use the category name. Identify it by its meaning.\n"
                "5. Include the clause number or section number when available.\n"
                "6. Do not classify a clause as absent from the entire document "
                "because it is missing from this page.\n"
                "7. Do not generate risk ratings or legal conclusions here.\n"
                "8. If no relevant clause is found, return "
                "'No matching clauses found in this page.'\n\n"

                "CRITICAL PAGE RULE:\n"
                "9. DO NOT generate page numbers.\n"
                "10. DO NOT mention page numbers.\n"
                "11. DO NOT guess or infer page numbers.\n"
                "12. The application already knows the verified PDF page number.\n"
                "13. Return ONLY the clause category and exact clause text.\n\n"

                "For every clause found, use exactly this format:\n"
                "✅ [CLAUSE CATEGORY] | Exact extracted text: [verbatim text]\n\n"

                f"DOCUMENT PAGE: {page_number}\n"
                f"PAGE SECTION {chunk_index} OF {len(text_chunks)}:\n"
                f"{text_chunk}\n"
            )

            result = _ask(
                prompt,
                max_tokens=1000,
            )

            if not result:
                continue

            for line in result.splitlines():

                stripped = line.strip()

                if not stripped:
                    continue

                if "No matching clauses found" in stripped:
                    continue

                # Keep only our expected clause format.
                if not stripped.startswith("✅"):
                    continue

                # Remove accidental page references if the model
                # ignores the instruction.
                cleaned_line = re.sub(
                    r"\s*\[Pages?\s+[^\]]+\]",
                    "",
                    stripped,
                    flags=re.IGNORECASE,
                )

                cleaned_line = re.sub(
                    r"\s*\(Pages?\s+\d+\)",
                    "",
                    cleaned_line,
                    flags=re.IGNORECASE,
                )

                # Do not allow an empty result.
                if not cleaned_line.strip():
                    continue

                # Deduplicate repeated extraction from the same page.
                dedup_key = (
                    str(page_number),
                    cleaned_line.lower(),
                )

                if not any(
                    existing_key == dedup_key
                    for existing_key, _ in all_results
                ):
                    all_results.append(
                        (
                            dedup_key,
                            f"{cleaned_line} | [Page {page_number}]",
                        )
                    )

    return "\n\n".join(
        result_text
        for _, result_text in all_results
    )

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
    "You are an AI legal risk analysis assistant for Tata Group documents.\n"
    "Your task is to identify and explain genuine legal, financial, "
    "contractual, governance, or compliance risks supported by the "
    "document clauses and, where applicable, approved Tata Group "
    "policy standards retrieved from the knowledge base.\n\n"

    f"DOCUMENT TYPE: {document_type}\n\n"

    "CORE OBJECTIVE:\n"
    "Analyze the extracted clauses themselves first and identify any "
    "material risk, exposure, obligation, authority, ambiguity, or "
    "potentially unfavorable provision that is actually supported by "
    "the document text.\n"
    "Then determine whether any applicable approved Tata Group policy "
    "standard supports a policy deviation or additional concern.\n\n"

    "IMPORTANT EVIDENCE RULES:\n"
    "1. Use only the extracted clauses and approved policy standards "
    "provided below as evidence.\n"
    "2. Do not use outside legal knowledge as factual evidence.\n"
    "3. Do not invent, modify, or assume missing contract language.\n"
    "4. Every risk finding must be directly traceable to an extracted "
    "clause present in the document.\n"
    "5. Do not create a risk merely because a clause is different from "
    "what is normally expected in a contract.\n"
    "6. Do not treat the absence of a clause as a risk unless an "
    "applicable approved Tata Group policy explicitly requires that "
    "clause for this document type.\n"
    "7. If the evidence does not support a risk, do not manufacture one.\n"
    "8. If policy evidence is unavailable or irrelevant, continue with "
    "clause-level risk analysis. Lack of a policy reference must NOT "
    "prevent identification of a genuine risk visible in the document.\n"
    "9. Never invent a Tata Group policy requirement, policy number, "
    "policy name, threshold, approval requirement, or deviation.\n"
    "10. Do not interpret an irrelevant policy match as evidence of risk.\n\n"

    "RISK IDENTIFICATION RULES:\n"
    "11. Identify a risk when a clause creates or permits a material "
    "legal, financial, contractual, governance, or compliance exposure.\n"
    "12. Broad or potentially unlimited indemnification obligations may "
    "be flagged when the extracted clause itself creates that exposure.\n"
    "13. Authority to provide guarantees, act as surety, assume debts, "
    "support third-party obligations, or enter into guarantees or "
    "indemnities may be flagged as potential financial exposure when "
    "such authority is expressly present in the document.\n"
    "14. Liability provisions must be interpreted according to the "
    "actual document context. Do not convert a general constitutional "
    "statement into a contractual liability cap.\n"
    "15. A clause may be classified as informational rather than risky "
    "when it does not create a material exposure.\n"
    "16. Do not inflate severity merely because a clause sounds broad. "
    "Severity must reflect the evidence and the potential significance "
    "of the exposure.\n"
    "17. When a clause creates potential exposure but the available "
    "evidence is insufficient to determine its exact financial or legal "
    "impact, identify the exposure and clearly state the limitation "
    "instead of guessing.\n\n"

    "MOA/AOA-SPECIFIC RULES:\n"
    "18. If the document type is MOA/AOA, do NOT treat the statement "
    "'the liability of the members is limited' as a vendor-contract "
    "limitation-of-liability cap.\n"
    "19. The members' limited liability statement may be reported as "
    "an informational constitutional provision, but it must not be "
    "presented as a vendor liability protection.\n"
    "20. If an MOA/AOA clause gives the company authority to act as "
    "guarantor or surety for debts, defaults, loans, or obligations "
    "of third parties, this may be identified as potential financial "
    "exposure because the authority is expressly present in the "
    "document.\n"
    "21. If an MOA/AOA clause provides indemnification to officers, "
    "directors, promoters, employees, consultants, or other persons, "
    "an appropriate risk may be identified when the wording creates "
    "broad financial or legal exposure.\n"
    "22. Do not apply vendor-contract requirements to MOA/AOA documents "
    "unless an approved policy reference explicitly states that the "
    "requirement applies to MOA/AOA.\n"
    "23. If a retrieved policy is specific to vendor contracts and the "
    "document is MOA/AOA, explicitly treat that policy as not directly "
    "applicable rather than reporting a policy deviation.\n\n"

    "POLICY ANALYSIS RULES:\n"
    "24. First determine whether each retrieved policy reference is "
    "actually applicable to the document type.\n"
    "25. Only report a policy deviation when the policy text explicitly "
    "supports the requirement and the document provides contrary "
    "evidence or fails to satisfy an explicitly applicable requirement.\n"
    "26. If no applicable policy standard is available, report the "
    "clause-level risk without inventing a policy deviation.\n"
    "27. Clearly distinguish between:\n"
    "   - Intrinsic document risk\n"
    "   - Policy deviation\n"
    "   - Informational observation\n"
    "   - Insufficient evidence\n\n"

    "SEVERITY GUIDANCE:\n"
    "Use HIGH when the clause creates a potentially significant or "
    "broad financial, legal, guarantee, indemnity, or other material "
    "exposure.\n"
    "Use MEDIUM when the clause creates a meaningful exposure that "
    "appears limited, conditional, or dependent on further facts.\n"
    "Use LOW when the concern is relatively minor, narrow, or mainly "
    "requires clarification or monitoring.\n"
    "Use the severity conservatively and only when supported by the "
    "actual clause.\n\n"

    "EXTRACTED CLAUSES ACTUALLY PRESENT IN THE DOCUMENT:\n"
    f"{retrieval_query}\n\n"

    "APPROVED TATA GROUP POLICY STANDARDS RETRIEVED FROM THE "
    "KNOWLEDGE BASE:\n"
    f"{reference_text}\n\n"

    "ANALYSIS PROCESS:\n"
    "Step 1: Review every extracted clause.\n"
    "Step 2: Identify clauses that create a legal, financial, "
    "contractual, governance, or compliance exposure.\n"
    "Step 3: Assign an appropriate severity only where justified.\n"
    "Step 4: Check whether an applicable approved Tata policy supports "
    "or contradicts the finding.\n"
    "Step 5: Do not create a policy deviation when the policy is "
    "irrelevant or not applicable.\n"
    "Step 6: Preserve the distinction between actual risk and "
    "informational observations.\n"
    "Step 7: If no genuine risk is supported by the evidence, clearly "
    "state that no material risk was identified.\n\n"

    "OUTPUT FORMAT:\n"
    "For every identified risk, return:\n"
    "- clause_type: the relevant extracted clause category\n"
    "- severity: low / medium / high\n"
    "- finding: concise description of the actual risk or exposure\n"
    "- rationale: explain exactly how the extracted clause creates "
    "the identified exposure\n"
    "- policy_reference: applicable policy reference if one exists; "
    "otherwise write 'No applicable policy reference identified'\n"
    "- risk_basis: 'Document clause' or 'Policy deviation'\n"
    "- page: use the page number already associated with the extracted "
    "clause; do not invent or guess page numbers\n\n"

    "IMPORTANT FINAL RULE:\n"
    "Do NOT return an empty risk analysis merely because no applicable "
    "policy reference was retrieved. If the document itself contains "
    "a clause that creates a genuine potential exposure, identify that "
    "exposure and classify it based on the evidence in the clause.\n"
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
