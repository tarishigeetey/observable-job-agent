"""Extract a structured candidate profile from CV text."""

from __future__ import annotations

from job_scout.config import get_settings
from job_scout.graph.schemas import Profile
from job_scout.llm import get_chat_model

EXTRACT_PROFILE_PROMPT_NAME = "extract_profile"

EXTRACT_PROFILE_PROMPT = """You are a recruiting assistant. Read the CV text below and extract a structured candidate profile.

Fill in every field:
- name: the candidate's name, or null if not present.
- seniority: one of junior, mid, senior, lead, or unknown.
- primary_roles: the job titles/roles this person is a fit for, ordered with their current or most recent role first.
- skills: a list of their skills, lowercased.
- years_experience: total years of professional experience as a number, or null.
- locations: locations where they could work.
- languages: spoken languages.
- remote_ok: true if they are open to remote work.
- raw_summary: a 3-4 sentence summary, starting with their most recent experience.

CV text:
{cv_text}
"""


def extract_profile(cv_text: str, *, thread_id: str | None = None, tags: list[str] | None = None) -> Profile:
    """Extract a structured profile from CV text with a single LLM call."""
    settings = get_settings()
    model = get_chat_model(settings.scout_model, temperature=0.0).with_structured_output(Profile)

    tracer = None
    if thread_id:
        from job_scout.tracing import get_tracer
        tracer = get_tracer(thread_id, tags or ["extract"])

    config = {"callbacks": [tracer]} if tracer else {}
    profile: Profile = model.invoke(EXTRACT_PROFILE_PROMPT.format(cv_text=cv_text), config=config)
    if tracer:
        tracer.flush()
    return profile
