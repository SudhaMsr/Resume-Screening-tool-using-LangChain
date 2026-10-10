"""
Claude screener: a LangChain prompt template feeding Claude's structured output.

LangChain formats the job + CV into messages; the Anthropic SDK's `messages.parse` returns a
validated CandidateAssessment (no JSON parsing or retries on malformed output needed).
"""

from __future__ import annotations

import anthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from .schema import CandidateAssessment

MODEL = "claude-opus-5"

SYSTEM = """You are an experienced technical recruiter screening CVs for a hiring manager.

Assess how well the CV fits the job description, using only evidence in the CV.
- Judge skills and experience, not wording: a skill counts if the CV shows it being used, even under another name.
- Be calibrated: most candidates are partial fits. Reserve 85+ for clear, well-evidenced matches.
- The CV has been anonymised. Never infer or mention age, gender, ethnicity, nationality, disability,
  family status or any other protected characteristic, and do not penalise career breaks.
- Keep strengths and concerns specific and tied to what the CV says."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "<job_description>\n{job}\n</job_description>\n\n<cv>\n{cv}\n</cv>\n\nAssess this candidate."),
])


class RefusedError(RuntimeError):
    """Claude declined to assess this CV."""


def build_chain(client: anthropic.Anthropic | None = None, model: str = MODEL) -> Runnable:
    """job + cv -> CandidateAssessment, as a LangChain runnable."""
    client = client or anthropic.Anthropic()

    def call_claude(prompt_value) -> CandidateAssessment:
        messages = prompt_value.to_messages()
        system = next(m.content for m in messages if m.type == "system")
        user = next(m.content for m in messages if m.type == "human")
        response = client.beta.messages.parse(
            model=model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=CandidateAssessment,
            output_config={"effort": "medium"},
            # if a safety classifier declines, re-run on a fallback model in the same call
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise RefusedError(f"Claude did not return an assessment (stop_reason={response.stop_reason})")
        return response.parsed_output

    return PROMPT | RunnableLambda(call_claude)
