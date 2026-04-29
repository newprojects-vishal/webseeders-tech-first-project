# DocQA — Local Document Q&A System

A lightweight, fully offline question-answering system that lets you upload a PDF and ask natural language questions about its content. All processing happens locally — no cloud APIs, no data leaves your machine.

## Why I Built It This Way

Most document Q&A tools rely on external APIs like OpenAI. I wanted to explore whether the same quality of experience could be achieved entirely on local hardware using open-source models. The answer is yes — with the right architecture.

The core idea is RAG (Retrieval-Augmented Generation): instead of asking a language model to memorize your document, you retrieve only the relevant parts at query time and feed them as context. This keeps answers grounded in the actual document.

## Architecture
PDF File
│
▼
┌──────────────┐
│   PyMuPDF    │  ← extracts text page by page
└──────┬───────┘
│
▼
┌──────────────┐
│   Chunker    │  ← splits into 600-token overlapping chunks
└──────┬───────┘
│
▼
┌────────────────────┐
│ sentence-transformers│ ← converts chunks to vector embeddings
└──────┬─────────────┘
│
▼
┌──────────────┐
│   ChromaDB   │  ← stores and indexes vectors locally
└──────┬───────┘
│
User asks a question
│
▼
┌────────────────────┐
│ sentence-transformers│ ← embeds the question
└──────┬─────────────┘
│
▼
┌──────────────┐
│   ChromaDB   │  ← finds top 4 most similar chunks
└──────┬───────┘
│
▼
┌──────────────┐
│  TinyLlama   │  ← generates answer using only retrieved context
│  via Ollama  │
└──────┬───────┘
│
▼
Answer + source pages shown in UI

## Tech Stack

| Layer | Library | Purpose |
|-------|---------|---------|
| UI | Streamlit | Web interface |
| PDF Parsing | PyMuPDF | Text extraction |
| Chunking | LangChain | Text splitting |
| Embeddings | sentence-transformers | Semantic vectors |
| Vector Store | ChromaDB | Local similarity search |
| LLM | TinyLlama via Ollama | Answer generation |

## Project Structure
doc-qa-system/
├── app.py                  # Streamlit UI
├── pipeline/
│   ├── init.py
│   ├── loader.py           # PDF text extraction
│   ├── chunker.py          # Text splitting
│   ├── embedder.py         # Embedding model
│   ├── vector_store.py     # ChromaDB operations
│   └── generator.py        # LLM answer generation
├── requirements.txt
└── README.md

## Prerequisites

- Python 3.10 or higher
- Ollama installed on your machine
- At least 2GB free RAM

## Setup

### 1. Install Ollama

Download and install from https://ollama.com/download

### 2. Pull the language model

```bash
ollama pull tinyllama
```

### 3. Clone the repository

```bash
git clone https://github.com/yourusername/doc-qa-system.git
cd doc-qa-system
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`

## Usage

1. Make sure Ollama is running in the background
2. Upload a PDF using the sidebar uploader
3. Wait for ingestion to complete (chunking + embedding)
4. Type your question in the input box
5. The system returns an answer with source page references

## Design Decisions

**Chunk size of 600 tokens with 80 token overlap** — large enough to preserve context within a passage, small enough that retrieved chunks stay focused on a single topic. Overlap prevents answers from being cut off at chunk boundaries.

**all-MiniLM-L6-v2 for embeddings** — fast, lightweight, and performs well on semantic similarity tasks. Runs entirely on CPU so no GPU required.

**ChromaDB for vector storage** — persistent local storage means a PDF only needs to be ingested once. Re-uploading the same file skips re-ingestion automatically.

**TinyLlama as the LLM** — chosen for its minimal RAM footprint (~600MB) while still being capable of coherent answer generation when given clear context.

## Limitations

- Answer quality depends on TinyLlama's capability — larger models will produce better answers if the machine has more RAM
- Scanned PDFs without embedded text layers are not supported
- Very large PDFs (100+ pages) may take longer to ingest on first upload
