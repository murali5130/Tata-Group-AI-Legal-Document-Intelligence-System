from guardrails.input_guardrail import _ask_guardrail


def output_guardrail(generated_response: str) -> bool:
    prompt = f"""
    Review this drafted response for highly sensitive information that should
    not be exposed from a legal contract, such as: government ID numbers
    (PAN, Aadhaar, SSN, passport numbers), bank account or IFSC details,
    unredacted signatory personal contact information, or confidential
    financial terms explicitly marked as restricted in the source document.

    Note: standard contract details such as company names, business addresses,
    monetary values in payment/rent clauses, dates, and clause text are NOT
    sensitive and should be treated as CLEAN.

    Drafted response:
    {generated_response}

    Reply with exactly one word: CLEAN or LEAK
    """
    try:
        raw_text = _ask_guardrail(prompt)
        return "CLEAN" in raw_text
    except Exception:
        return False
