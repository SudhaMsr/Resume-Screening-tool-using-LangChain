"""LangChain document loaders for CVs: PDF, DOCX, TXT and Markdown."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import docx2txt
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def extract_text(data: bytes, filename: str) -> str:
    """Plain text from a CV file's bytes, chosen by extension."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if suffix == ".docx":
        return docx2txt.process(io.BytesIO(data)).strip()
    if suffix in {".txt", ".md"}:
        return data.decode("utf-8", errors="replace").strip()
    raise ValueError(f"Unsupported file type '{suffix}'. Use one of: {', '.join(sorted(SUPPORTED))}")


class ResumeLoader(BaseLoader):
    """Loads every supported CV in a folder (or a single file) as one Document each."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def lazy_load(self) -> Iterator[Document]:
        files = [self.path] if self.path.is_file() else sorted(p for p in self.path.iterdir() if p.suffix.lower() in SUPPORTED)
        for f in files:
            text = extract_text(f.read_bytes(), f.name)
            if text:
                yield Document(page_content=text, metadata={"source": f.name})


class UploadedResumeLoader(BaseLoader):
    """Same as ResumeLoader, for in-memory uploads such as Streamlit's file_uploader."""

    def __init__(self, files: list[tuple[str, bytes]]):
        self.files = files

    def lazy_load(self) -> Iterator[Document]:
        for name, data in self.files:
            text = extract_text(data, name)
            if text:
                yield Document(page_content=text, metadata={"source": name})
