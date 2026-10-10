"""Streamlit UI:  streamlit run app.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from resume_screener import ResumeLoader, UploadedResumeLoader, screen

SAMPLES = Path(__file__).parent / "samples"
LABEL = {"strong_yes": "Strong yes", "yes": "Yes", "maybe": "Maybe", "no": "No"}

st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="wide")
st.title("Resume Screener")
st.caption("Rank CVs against a job description. CVs are anonymised before scoring; a person should make every hiring decision.")

with st.sidebar:
    st.header("Settings")
    use_samples = st.checkbox("Use the sample job and CVs", value=True)
    st.caption("Runs entirely offline: skill matching + TF-IDF. No API key needed.")

left, right = st.columns([1, 1])
with left:
    default_job = (SAMPLES / "job_description.md").read_text() if use_samples else ""
    job = st.text_area("Job description", default_job, height=320)
with right:
    uploads = st.file_uploader("CVs (PDF, DOCX, TXT, MD)", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if use_samples and not uploads:
        st.caption(f"Using {len(list((SAMPLES / 'resumes').iterdir()))} sample CVs from `samples/resumes`.")

if st.button("Rank candidates", type="primary", disabled=not job.strip()):
    docs = UploadedResumeLoader([(f.name, f.getvalue()) for f in uploads]).load() if uploads else (
        ResumeLoader(SAMPLES / "resumes").load() if use_samples else [])
    if not docs:
        st.warning("Add at least one CV.")
        st.stop()

    ranking = screen(docs, job)

    table = pd.DataFrame([{
        "Rank": i, "Candidate": r.candidate_id, "Score": r.assessment.match_score,
        "Verdict": LABEL[r.assessment.recommendation], "File": r.source,
        "Missing": ", ".join(r.assessment.missing_skills) or "–",
    } for i, r in enumerate(ranking, 1)])
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d")})
    st.download_button("Download CSV", table.to_csv(index=False), "ranking.csv", "text/csv")

    st.subheader("Details")
    for r in ranking:
        a = r.assessment
        with st.expander(f"{r.candidate_id} · {a.match_score} · {LABEL[a.recommendation]}  ({r.source})"):
            st.write(a.summary)
            c1, c2 = st.columns(2)
            c1.markdown("**Matched skills**\n\n" + (", ".join(a.matched_skills) or "–"))
            c2.markdown("**Missing skills**\n\n" + (", ".join(a.missing_skills) or "–"))
            c1.markdown("**Strengths**\n" + "".join(f"\n- {s}" for s in a.strengths))
            c2.markdown("**Concerns**\n" + "".join(f"\n- {s}" for s in a.concerns))
            if a.years_experience is not None:
                st.caption(f"Relevant experience: about {a.years_experience:g} years")
