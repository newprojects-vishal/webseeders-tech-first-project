from typing import Dict, List

import fitz


def load_pdf(file_path: str) -> List[Dict[str, object]]:
    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        raise ValueError(f"Unable to open PDF: {exc}") from exc
    if doc.needs_pass:
        doc.close()
        raise ValueError("The PDF is encrypted and cannot be processed.")
    pages: List[Dict[str, object]] = []
    try:
        for idx, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({"page": idx, "text": text})
    except Exception as exc:
        raise ValueError(f"Unable to read PDF content: {exc}") from exc
    finally:
        doc.close()
    return pages
