"""
Offline screener: no API key, no network.

Scores a CV by (a) how many of the job's required and preferred skills it evidences and
(b) TF-IDF similarity to the job description. Used on its own for a free first pass, and
by the pipeline to shortlist CVs before sending only the best ones to Claude.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import CandidateAssessment

# canonical skill -> patterns that count as evidence for it
SKILLS: dict[str, list[str]] = {
    "Python": [r"\bpython\b"], "SQL": [r"\bsql\b", r"postgres", r"mysql", r"bigquery"], "R": [r"\bR\b(?= |,|/|\.)"],
    "Machine learning": [r"machine learning", r"\bml\b", r"\bmodel(s|ling|ing)?\b", r"classifiers?"], "Deep learning": [r"deep learning", r"neural network"],
    "PyTorch": [r"pytorch", r"\btorch\b"], "TensorFlow": [r"tensorflow", r"\bkeras\b"], "scikit-learn": [r"scikit-learn", r"sklearn"],
    "XGBoost": [r"xgboost", r"lightgbm", r"gradient boost"], "NLP": [r"\bnlp\b", r"natural language", r"transformers?\b", r"\bbert\b"],
    "LLMs": [r"\bllms?\b", r"large language model", r"\bgpt\b", r"claude", r"generative ai", r"genai"],
    "RAG": [r"\brag\b", r"retrieval[- ]augmented", r"vector (store|database)", r"embeddings"],
    "LangChain": [r"langchain", r"llamaindex"], "Computer vision": [r"computer vision", r"opencv", r"\byolo\b", r"image classification"],
    "Statistics": [r"statistic", r"hypothesis test", r"a/b test", r"regression", r"bayesian"],
    "Time series": [r"time[- ]series", r"forecast"], "Data visualisation": [r"tableau", r"power ?bi", r"matplotlib", r"seaborn", r"plotly", r"visuali[sz]"],
    "pandas": [r"\bpandas\b", r"numpy"], "Spark": [r"\bspark\b", r"pyspark", r"databricks"],
    "Cloud": [r"\baws\b", r"\bgcp\b", r"google cloud", r"\bazure\b", r"sagemaker", r"vertex ai"],
    "Docker": [r"docker", r"kubernetes", r"\bk8s\b"], "MLOps": [r"mlops", r"mlflow", r"model monitoring", r"ci/cd", r"airflow"],
    "APIs": [r"fastapi", r"\bflask\b", r"rest api", r"\bapis?\b"], "Git": [r"\bgit\b", r"github"],
    "Model optimisation": [r"quanti[sz]", r"model compression", r"pruning", r"distillation", r"onnx"],
    "Communication": [r"stakeholder", r"present(ed|ing|ation)", r"communicat"],
}

REQUIRED_HEAD = re.compile(r"(required|requirements|must[- ]have|essential|what you.ll need|you have)", re.I)
PREFERRED_HEAD = re.compile(r"(preferred|nice[- ]to[- ]have|bonus|desirable|plus)", re.I)
YEARS = re.compile(r"(\d{1,2})\+?\s*(?:years|yrs)", re.I)


def find_skills(text: str) -> set[str]:
    return {skill for skill, pats in SKILLS.items() if any(re.search(p, text, re.I) for p in pats)}


Group = frozenset[str]  # a requirement met by any one of its skills ("PyTorch or TensorFlow")


def split_job_skills(jd: str) -> tuple[list[Group], list[Group]]:
    """(required, preferred) requirement groups, read from the job's section headings when present."""
    required: list[Group] = []
    preferred: list[Group] = []
    mode = "required"
    for line in jd.splitlines():
        if line.strip().startswith("#") or line.strip().endswith(":") or len(line.split()) <= 5:
            if PREFERRED_HEAD.search(line):
                mode = "preferred"
            elif REQUIRED_HEAD.search(line):
                mode = "required"
        found = find_skills(line)
        if not found:
            continue
        # "X or Y" on one line is a single either/or requirement; otherwise each skill stands alone
        groups = [frozenset(found)] if re.search(r"\bor\b|/", line, re.I) and len(found) > 1 else [frozenset({f}) for f in found]
        target = preferred if mode == "preferred" else required
        target.extend(g for g in groups if g not in target)
    if not required:
        required = [frozenset({s}) for s in find_skills(jd)]
    return required, [g for g in preferred if g not in required]


def _label(group: Group) -> str:
    return " or ".join(sorted(group))


def years_of_experience(text: str) -> float | None:
    hits = [int(m) for m in YEARS.findall(text) if int(m) < 45]
    return float(max(hits)) if hits else None


def assess(cv: str, jd: str) -> CandidateAssessment:
    required, preferred = split_job_skills(jd)
    have = find_skills(cv)
    req_met = [g for g in required if g & have]
    pref_met = [g for g in preferred if g & have]
    req_cov = len(req_met) / len(required) if required else 0.0
    pref_cov = len(pref_met) / len(preferred) if preferred else 0.0

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit([jd, cv])
    sim = float(cosine_similarity(tfidf.transform([jd]), tfidf.transform([cv]))[0, 0])

    score = round(100 * (0.6 * req_cov + 0.15 * pref_cov + 0.25 * min(1.0, sim / 0.35)))
    rec = "strong_yes" if score >= 85 else "yes" if score >= 70 else "maybe" if score >= 50 else "no"
    missing = sorted(_label(g) for g in required if not g & have)
    matched = sorted({s for g in req_met + pref_met for s in g & have})
    return CandidateAssessment(
        match_score=score,
        recommendation=rec,
        summary=(
            f"Meets {len(req_met)} of {len(required)} required skills"
            + (f" and {len(pref_met)} of {len(preferred)} nice-to-haves" if preferred else "")
            + f"; text similarity to the job is {sim:.2f}. Keyword-based estimate, so review before deciding."
        ),
        matched_skills=matched,
        missing_skills=missing,
        strengths=[f"Evidence of {s}" for s in sorted({s for g in req_met for s in g & have})[:3]],
        concerns=[f"No evidence of {m}" for m in missing[:3]],
        years_experience=years_of_experience(cv),
    )
