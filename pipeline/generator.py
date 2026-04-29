from typing import Dict, List

import requests


def generate_answer(question: str, context_chunks: List[Dict[str, object]]) -> str:
    context = "\n\n".join(
        [f"[Page {item['page']}]\n{item['text']}" for item in context_chunks]
    )
    system = (
        "You answer questions using only the provided context. "
        "Always cite page numbers in your answer. "
        "If the answer is not present, reply exactly: "
        '"The document does not contain information about this."'
    )
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "tinyllama", "prompt": prompt, "system": system, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return str(response.json().get("response", "")).strip()
