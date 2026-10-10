"""
Command line:

    python -m resume_screener --job samples/job_description.md --resumes samples/resumes
    python -m resume_screener --job jd.md --resumes cvs/ --csv ranking.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .loaders import ResumeLoader
from .pipeline import screen

LABEL = {"strong_yes": "Strong yes", "yes": "Yes", "maybe": "Maybe", "no": "No"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="resume_screener", description="Rank CVs against a job description.")
    ap.add_argument("--job", required=True, help="job description file (.md or .txt)")
    ap.add_argument("--resumes", required=True, help="folder of CVs (.pdf, .docx, .txt, .md)")
    ap.add_argument("--csv", help="also write the ranking to this CSV file")
    args = ap.parse_args(argv)

    job = Path(args.job).read_text(encoding="utf-8")
    docs = ResumeLoader(args.resumes).load()
    if not docs:
        print(f"No CVs found in {args.resumes}", file=sys.stderr)
        return 1

    ranking = screen(docs, job)

    print(f"\n{'#':>2}  {'Candidate':<13} {'Score':>5}  {'Verdict':<10} File")
    print("-" * 60)
    for i, r in enumerate(ranking, 1):
        a = r.assessment
        print(f"{i:>2}  {r.candidate_id:<13} {a.match_score:>5}  {LABEL[a.recommendation]:<10} {r.source}")
    best = ranking[0].assessment
    print(f"\nTop candidate: {best.summary}")
    if best.missing_skills:
        print(f"Gaps: {', '.join(best.missing_skills)}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "candidate", "file", "score", "recommendation", "matched_skills", "missing_skills", "summary"])
            for i, r in enumerate(ranking, 1):
                a = r.assessment
                w.writerow([i, r.candidate_id, r.source, a.match_score, a.recommendation,
                            "; ".join(a.matched_skills), "; ".join(a.missing_skills), a.summary])
        print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
