import re
from typing import Dict, List

import chromadb
from langchain_core.documents import Document

_CLIENT = chromadb.PersistentClient(path="./chroma_db")


def get_collection_name(filename: str) -> str:
    name = filename.rsplit(".", 1)[0].lower()
    clean = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return clean or "document"


def store_chunks(chunks: List[Document], embeddings: List[List[float]], collection_name: str) -> bool:
    collection = _CLIENT.get_or_create_collection(name=collection_name)
    if collection.count() > 0:
        return False
    collection.add(
        ids=[f"{collection_name}_{i}" for i in range(len(chunks))],
        documents=[doc.page_content for doc in chunks],
        metadatas=[doc.metadata for doc in chunks],
        embeddings=embeddings,
    )
    return True


def query_store(
    query_embedding: List[float], collection_name: str, n_results: int = 4
) -> List[Dict[str, object]]:
    collection = _CLIENT.get_or_create_collection(name=collection_name)
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return [
        {"text": docs[i], "page": int(metas[i].get("page", 0)), "distance": float(dists[i])}
        for i in range(len(docs))
    ]
