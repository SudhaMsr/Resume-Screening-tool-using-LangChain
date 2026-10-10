"""
The screening pipeline (fully offline, no API key):

    load CVs -> anonymise -> score (required skills, nice-to-haves, TF-IDF) -> rank
"""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from . import offline
from .anonymize import anonymize
from .schema import RankedCandidate


def screen(docs: Iterable[Document], job: str) -> list[RankedCandidate]:
    """Rank CVs for a job, best first."""
    # LangChain runnable: Document -> anonymised Document
    blind = RunnableLambda(anonymize).batch(list(docs))
    ranked = [
        RankedCandidate(candidate_id=f"Candidate {i + 1:02d}", source=d.metadata["source"],
                        assessment=offline.assess(d.page_content, job))
        for i, d in enumerate(blind)
    ]
    return sorted(ranked, key=lambda r: r.assessment.match_score, reverse=True)
