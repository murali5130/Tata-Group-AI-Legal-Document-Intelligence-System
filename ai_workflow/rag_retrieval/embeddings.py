from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 384  # MUST match the vector(384) column in the Supabase table

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> list[float]:
    """Returns a 384-dimensional embedding for the given text."""
    return _get_model().encode(text).tolist()