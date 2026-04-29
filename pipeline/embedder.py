from typing import List

from sentence_transformers import SentenceTransformer

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors = _get_model().encode(texts, convert_to_numpy=True)
    return [vector.tolist() for vector in vectors]


def embed_query(query: str) -> List[float]:
    vector = _get_model().encode(query, convert_to_numpy=True)
    return vector.tolist()
