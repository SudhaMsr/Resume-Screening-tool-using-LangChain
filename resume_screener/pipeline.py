"""
The screening pipeline:

    load CVs -> anonymise -> offline score (all) -> shortlist top N -> Claude assessment -> rank

Claude only sees the shortlist, which keeps cost proportional to N rather than to the
size of the applicant pool.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from . import offline
from .anonymize import anonymize
from .claude import RefusedError
from .schema import RankedCandidate


def screen(
    docs: Iterable[Document],
    job: str,
    *,
    claude_chain: Runnable | None = None,
    shortlist: int = 5,
    on_progress: Callable[[str], None] | None = None,
) -> list[RankedCandidate]:
    """Rank CVs for a job. With no claude_chain, everything is scored offline."""
    say = on_progress or (lambda _: None)
    blind = [anonymize(d) for d in docs]
    ids = {d.metadata["source"]: f"Candidate {i + 1:02d}" for i, d in enumerate(blind)}

    first_pass = sorted(
        (
            RankedCandidate(candidate_id=ids[d.metadata["source"]], source=d.metadata["source"],
                            assessment=offline.assess(d.page_content, job), method="offline")
            for d in blind
        ),
        key=lambda r: r.assessment.match_score,
        reverse=True,
    )
    if claude_chain is None:
        return first_pass

    by_source = {d.metadata["source"]: d for d in blind}
    top, rest = first_pass[:shortlist], first_pass[shortlist:]
    results: list[RankedCandidate] = []
    for i, r in enumerate(top, 1):
        say(f"Claude is assessing {r.candidate_id} ({i}/{len(top)})")
        try:
            assessment = claude_chain.invoke({"job": job, "cv": by_source[r.source].page_content})
            results.append(r.model_copy(update={"assessment": assessment, "method": "claude"}))
        except RefusedError:
            results.append(r)  # keep the offline score rather than drop the candidate
    results.sort(key=lambda r: r.assessment.match_score, reverse=True)
    return results + rest
