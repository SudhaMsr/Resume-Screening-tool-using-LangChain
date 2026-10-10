"""
Blind screening: strip direct identifiers before a CV is scored.

Removes emails, phone numbers, URLs, the name line at the top of the CV, and common
personal fields (date of birth, gender, nationality, marital status, photo). This reduces
the chance that a score is influenced by who the candidate is rather than what they can do.
It is a mitigation, not a guarantee: names can still appear inside free text.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL = re.compile(r"(https?://\S+|www\.\S+|\b(?:linkedin|github)\.com/\S+)", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
PERSONAL = re.compile(
    r"^\s*(date of birth|dob|age|gender|sex|nationality|marital status|religion|photo)\s*[:\-].*$",
    re.I | re.M,
)


def _looks_like_name(line: str) -> bool:
    words = line.split()
    return 1 < len(words) <= 4 and all(w[:1].isupper() and w.replace("-", "").replace("'", "").isalpha() for w in words)


def anonymize_text(text: str) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    text = URL.sub("[LINK]", text)
    text = PHONE.sub("[PHONE]", text)
    text = PERSONAL.sub("[PERSONAL DETAIL REMOVED]", text)
    lines = text.splitlines()
    # the first non-empty line of a CV is almost always the candidate's name
    for i, line in enumerate(lines):
        if line.strip():
            if _looks_like_name(line.strip().lstrip("# ").strip()):
                lines[i] = "[CANDIDATE]"
            break
    return "\n".join(lines)


def anonymize(doc: Document) -> Document:
    return Document(page_content=anonymize_text(doc.page_content), metadata={**doc.metadata, "anonymized": True})
