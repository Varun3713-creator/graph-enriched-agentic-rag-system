"""
rag/document_loader.py — Format detection and raw text extraction
"""
from __future__ import annotations
import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def load_document(path: str) -> Tuple[str, str]:
    """
    Load a document and return (raw_text, format).
    Supported formats: pdf, html, txt
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _load_pdf(path), "pdf"
    elif ext in (".html", ".htm"):
        return _load_html(path), "html"
    else:
        return _load_text(path), "text"


def _load_pdf(path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required for PDF loading: pip install pdfplumber")

    pages: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    logger.info(f"Loaded PDF '{path}' — {len(pages)} pages")
    return "\n\n".join(pages)


def _load_html(path: str) -> str:
    from bs4 import BeautifulSoup
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    text = soup.get_text(separator="\n")
    logger.info(f"Loaded HTML '{path}'")
    return text


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    logger.info(f"Loaded text '{path}'")
    return text


def load_directory(directory: str) -> List[Tuple[str, str, str]]:
    """
    Load all supported docs from a directory.
    Returns list of (filename, raw_text, format).
    """
    supported = {".pdf", ".html", ".htm", ".txt", ".md"}
    results = []
    for fname in os.listdir(directory):
        ext = os.path.splitext(fname)[1].lower()
        if ext in supported:
            fpath = os.path.join(directory, fname)
            try:
                text, fmt = load_document(fpath)
                results.append((fname, text, fmt))
            except Exception as e:
                logger.error(f"Failed to load '{fname}': {e}")
    return results
