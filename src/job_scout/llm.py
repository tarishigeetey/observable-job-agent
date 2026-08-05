"""Chat-model factory and the per-run LLM call budget."""

from __future__ import annotations

import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from job_scout.config import get_settings


class LLMBudgetExceededError(RuntimeError):
    """Raised when a run would exceed MAX_LLM_CALLS_PER_RUN."""


def _export_openai_key() -> None:
    """Copy the OpenAI key from settings into the environment for LangChain."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    key = get_settings().openai_api_key.get_secret_value()
    if key:
        os.environ["OPENAI_API_KEY"] = key


def _export_groq_key() -> None:
    """Copy the Groq key from settings into the environment for LangChain."""
    if os.environ.get("GROQ_API_KEY"):
        return
    key = get_settings().groq_api_key.get_secret_value()
    if key:
        os.environ["GROQ_API_KEY"] = key


@lru_cache(maxsize=8)
def get_chat_model(model: str, temperature: float = 0.0) -> BaseChatModel:
    """Return a cached chat model for a LangChain provider string (e.g. openai:gpt-4o-mini)."""
    if model.startswith("openai:"):
        _export_openai_key()
    elif model.startswith("groq:"):
        _export_groq_key()
    return init_chat_model(model, temperature=temperature)


def ensure_budget(current_calls: int, planned: int, max_calls: int) -> None:
    """Raise LLMBudgetExceededError if planned more calls would exceed max_calls."""
    if current_calls + planned > max_calls:
        raise LLMBudgetExceededError(
            f"Run would make {current_calls + planned} LLM calls, exceeding MAX_LLM_CALLS_PER_RUN={max_calls}."
        )
