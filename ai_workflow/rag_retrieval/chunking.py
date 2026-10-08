def chunk_text(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[dict]:
    """
    Splits page-wise text into overlapping chunks.

    Input:
        [
            {"page_number": 1, "text": "..."},
            {"page_number": 2, "text": "..."}
        ]

    Output:
        [
            {
                "text": "...",
                "page_number": 1
            },
            {
                "text": "...",
                "page_number": 2
            }
        ]
    """

    chunks = []

    for page in pages:
        text = page.get("text", "").strip()
        page_number = page.get("page_number")

        if not text:
            continue

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "text": chunk,
                    "page_number": page_number
                })

            start += chunk_size - overlap

    return chunks