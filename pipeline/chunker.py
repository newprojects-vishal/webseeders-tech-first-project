from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(pages: List[Dict[str, object]], source: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    docs: List[Document] = []
    for item in pages:
        page_num = int(item["page"])
        text = str(item["text"])
        for chunk in splitter.split_text(text):
            docs.append(
                Document(page_content=chunk, metadata={"page": page_num, "source": source})
            )
    return docs
