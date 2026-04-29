from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

import requests
import streamlit as st

from pipeline import (
    create_chunks,
    embed_query,
    embed_texts,
    generate_answer,
    get_collection_name,
    load_pdf,
    query_store,
    store_chunks,
)


def check_ollama() -> bool:
    try:
        resp = requests.get("http://localhost:11434", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def init_state() -> None:
    st.session_state.setdefault("qa_history", [])
    st.session_state.setdefault("active_collection", None)
    st.session_state.setdefault("doc_info", None)


st.set_page_config(page_title="DocQA", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background-color: #0f1117; color: #e8eef4;}
    .stTextInput > div > div > input {background-color: #18202d; color: #e8eef4;}
    .answer-card {background: #161c27; border: 1px solid #233042; border-radius: 10px; padding: 14px; margin-bottom: 10px;}
    .src-tag {display: inline-block; background: #00b4a6; color: #001614; border-radius: 12px; padding: 2px 9px; margin-right: 6px; font-size: 12px;}
    .input-wrap {position: sticky; bottom: 0; background: #0f1117; padding-top: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

init_state()
ollama_ok = check_ollama()
st.title("DocQA")
if not ollama_ok:
    st.error("Ollama is not running. Start it with `ollama serve` and pull the model using `ollama pull llama3.2:3b`.")

with st.sidebar:
    st.header("Setup")
    st.write(f"Ollama status: {'🟢 Running' if ollama_ok else '🔴 Offline'}")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        file_name = uploaded_file.name
        collection_name = get_collection_name(file_name)
        already_loaded = st.session_state["active_collection"] == collection_name
        if not already_loaded:
            try:
                with st.spinner("Ingesting document..."):
                    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                        temp.write(uploaded_file.getbuffer())
                        temp_path = temp.name
                    pages = load_pdf(temp_path)
                    chunks = create_chunks(pages, file_name)
                    vectors = embed_texts([doc.page_content for doc in chunks]) if chunks else []
                    was_ingested = store_chunks(chunks, vectors, collection_name) if chunks else False
                    Path(temp_path).unlink(missing_ok=True)
                st.session_state["active_collection"] = collection_name
                st.session_state["doc_info"] = {
                    "filename": file_name,
                    "pages": len(pages),
                    "chunks": len(chunks),
                    "status": "No extractable text found" if not chunks else ("Existing index reused" if not was_ingested else "Document indexed"),
                }
            except ValueError as exc:
                st.error(str(exc))
    if st.session_state["doc_info"]:
        info = st.session_state["doc_info"]
        st.success(str(info["status"]))
        st.write(f"File: {info['filename']}")
        st.write(f"Pages: {info['pages']}")
        st.write(f"Chunks: {info['chunks']}")

for item in st.session_state["qa_history"]:
    st.markdown(f"**You:** {item['question']}")
    st.markdown(f"<div class='answer-card'>{item['answer']}</div>", unsafe_allow_html=True)
    tags = "".join([f"<span class='src-tag'>Page {p}</span>" for p in item["pages"]])
    st.markdown(tags, unsafe_allow_html=True)

st.markdown("<div class='input-wrap'>", unsafe_allow_html=True)
with st.form("qa_form", clear_on_submit=False):
    question = st.text_input("Ask a question about your document")
    c1, c2 = st.columns([1, 1])
    ask = c1.form_submit_button("Ask", use_container_width=True, type="primary")
    clear = c2.form_submit_button("Clear chat", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if clear:
    st.session_state["qa_history"] = []
    st.rerun()

if ask and question.strip():
    if not ollama_ok:
        st.warning("Start Ollama before asking questions.")
    elif not st.session_state["active_collection"]:
        st.warning("Upload and ingest a PDF first.")
    else:
        with st.spinner("Generating answer..."):
            q_vec = embed_query(question.strip())
            results = query_store(q_vec, str(st.session_state["active_collection"]), n_results=4)
            answer = generate_answer(question.strip(), results)
        pages = sorted({int(x["page"]) for x in results if int(x["page"]) > 0})
        st.session_state["qa_history"].append(
            {"question": question.strip(), "answer": answer, "pages": pages}
        )
        st.rerun()
