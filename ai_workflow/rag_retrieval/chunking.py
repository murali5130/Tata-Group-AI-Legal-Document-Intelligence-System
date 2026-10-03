"""
ai_workflow/rag_retrieval/chunking.py

Splits a long document's text into smaller overlapping pieces before
embedding. Embedding an entire contract as one vector loses precision —
a question about "termination" should match the termination paragraph
specifically, not get diluted by every other paragraph in the contract.
"""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Simple character-based chunker with overlap. overlap exists so a
    clause that happens to span a chunk boundary isn't cut in a way that
    loses meaning — the tail of one chunk repeats as the head of the next.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks
