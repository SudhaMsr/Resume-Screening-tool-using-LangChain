from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from resume_screener import ResumeLoader, screen
from resume_screener.anonymize import anonymize_text
from resume_screener.claude import RefusedError, build_chain
from resume_screener.offline import assess, split_job_skills
from resume_screener.schema import CandidateAssessment

SAMPLES = Path(__file__).parent.parent / "samples"
JOB = (SAMPLES / "job_description.md").read_text()


def fake_assessment(score: int) -> CandidateAssessment:
    return CandidateAssessment(match_score=score, recommendation="yes", summary="ok", matched_skills=["Python"],
                               missing_skills=[], strengths=["s"], concerns=[], years_experience=3)


# ── anonymisation

def test_anonymize_removes_identifiers():
    cv = "Jane Example\njane@example.com | +44 7700 900123 | linkedin.com/in/jane\nDate of birth: 01/01/1990\nPython developer"
    out = anonymize_text(cv)
    assert "Jane" not in out and "jane@" not in out and "7700" not in out and "linkedin" not in out and "1990" not in out
    assert out.splitlines()[0] == "[CANDIDATE]"
    assert "Python developer" in out


def test_anonymize_keeps_non_name_first_line():
    assert anonymize_text("Senior Data Scientist with 7 years").startswith("Senior Data Scientist")


# ── loaders

def test_loader_reads_all_formats():
    docs = ResumeLoader(SAMPLES / "resumes").load()
    assert {Path(d.metadata["source"]).suffix for d in docs} == {".md", ".txt", ".docx"}
    assert all(d.page_content for d in docs)


# ── offline scoring

def test_either_or_requirement_is_one_group():
    required, preferred = split_job_skills(JOB)
    assert frozenset({"Deep learning", "PyTorch", "TensorFlow"}) in required
    assert all(g not in required for g in preferred)


def test_offline_ranks_relevant_cv_above_unrelated():
    strong = assess((SAMPLES / "resumes" / "amara_okafor.md").read_text(), JOB)
    weak = assess((SAMPLES / "resumes" / "sofia_martins.txt").read_text(), JOB)
    assert strong.match_score > weak.match_score + 40
    assert 0 <= weak.match_score <= 100 and strong.recommendation in {"yes", "strong_yes"}


# ── Claude chain (mocked: no network)

def mocked_client(parsed, stop_reason="end_turn"):
    client = MagicMock()
    client.beta.messages.parse.return_value = SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason)
    return client


def test_claude_chain_sends_anonymised_cv_and_schema():
    client = mocked_client(fake_assessment(88))
    result = build_chain(client).invoke({"job": "Need Python", "cv": "[CANDIDATE]\nPython for 5 years"})
    assert result.match_score == 88
    kwargs = client.beta.messages.parse.call_args.kwargs
    assert kwargs["output_format"] is CandidateAssessment
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["fallbacks"] == "default"
    assert "<cv>\n[CANDIDATE]\nPython for 5 years\n</cv>" in kwargs["messages"][0]["content"]
    assert "protected characteristic" in kwargs["system"]


def test_claude_refusal_raises():
    with pytest.raises(RefusedError):
        build_chain(mocked_client(None, "refusal")).invoke({"job": "j", "cv": "c"})


# ── pipeline

def test_pipeline_only_sends_shortlist_to_claude():
    docs = ResumeLoader(SAMPLES / "resumes").load()
    chain = MagicMock()
    chain.invoke.return_value = fake_assessment(99)
    ranking = screen(docs, JOB, claude_chain=chain, shortlist=2)
    assert chain.invoke.call_count == 2
    assert [r.method for r in ranking].count("claude") == 2
    assert len(ranking) == len(docs)
    assert ranking[0].assessment.match_score == 99
    # Claude never sees names or emails
    for call in chain.invoke.call_args_list:
        assert "@example" not in call.args[0]["cv"]


def test_pipeline_keeps_offline_score_when_claude_refuses():
    chain = MagicMock()
    chain.invoke.side_effect = RefusedError("declined")
    docs = [Document(page_content="Python SQL pandas machine learning", metadata={"source": "a.txt"})]
    ranking = screen(docs, JOB, claude_chain=chain, shortlist=1)
    assert ranking[0].method == "offline"
