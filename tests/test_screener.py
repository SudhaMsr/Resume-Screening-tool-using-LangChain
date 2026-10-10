from pathlib import Path

from resume_screener import ResumeLoader, screen
from resume_screener.anonymize import anonymize_text
from resume_screener.offline import assess, split_job_skills

SAMPLES = Path(__file__).parent.parent / "samples"
JOB = (SAMPLES / "job_description.md").read_text()


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


# ── pipeline

def test_pipeline_ranks_and_anonymises():
    docs = ResumeLoader(SAMPLES / "resumes").load()
    ranking = screen(docs, JOB)
    assert len(ranking) == len(docs)
    scores = [r.assessment.match_score for r in ranking]
    assert scores == sorted(scores, reverse=True)
    assert ranking[0].source == "amara_okafor.md" and ranking[-1].source == "sofia_martins.txt"
    assert all(r.candidate_id.startswith("Candidate ") for r in ranking)
