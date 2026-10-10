"""The structured assessment every screener (Claude or offline) returns for one CV."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Recommendation = Literal["strong_yes", "yes", "maybe", "no"]


class CandidateAssessment(BaseModel):
    match_score: int = Field(description="Overall fit for the role, 0-100. Anchor 85+ for exceptional, 70-84 strong, 50-69 partial, <50 weak.")
    recommendation: Recommendation = Field(description="strong_yes, yes, maybe or no")
    summary: str = Field(description="Two sentences on fit, written for a hiring manager.")
    matched_skills: list[str] = Field(description="Required or preferred skills from the job that the CV clearly evidences.")
    missing_skills: list[str] = Field(description="Required skills from the job with no evidence in the CV.")
    strengths: list[str] = Field(description="Up to 3 specific strengths, each tied to evidence in the CV.")
    concerns: list[str] = Field(description="Up to 3 specific gaps or risks. Never mention protected characteristics.")
    years_experience: float | None = Field(description="Relevant professional experience in years, if stated or inferable; otherwise null.")


class RankedCandidate(BaseModel):
    """One row of the final ranking."""

    candidate_id: str
    source: str
    assessment: CandidateAssessment
    method: Literal["claude", "offline"]
