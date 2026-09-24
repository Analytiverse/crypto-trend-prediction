"""
AlphaPulse Groq infrastructure client.

Owns the technical integration with the Groq API.
Application services should use this module instead of
creating Groq clients directly.

A request-level LLM call budget prevents a single chatbot
message from making more than the configured maximum number
of Groq API calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq

from src.core.config import ENV_FILE


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(
    dotenv_path=ENV_FILE
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY was not found. "
        "Add it to the project .env file."
    )


# ============================================================
# CONFIGURATION
# ============================================================

LLM_MODEL = "openai/gpt-oss-120b"

MAX_LLM_CALLS_PER_MESSAGE = 3

_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# LLM CALL BUDGET
# ============================================================

@dataclass
class LLMCallBudget:
    """
    Track Groq API calls made while processing one user message.
    """

    max_calls: int = MAX_LLM_CALLS_PER_MESSAGE
    calls_used: int = 0

    def consume(self) -> None:
        """
        Consume one available LLM call.

        Raise an error before the Groq request if the maximum
        number of calls has already been reached.
        """

        if self.calls_used >= self.max_calls:
            raise RuntimeError(
                "Maximum LLM API calls per user message exceeded."
            )

        self.calls_used += 1

    @property
    def calls_remaining(self) -> int:
        """
        Return the number of LLM calls still available.
        """

        return max(
            0,
            self.max_calls - self.calls_used,
        )


# ============================================================
# GUARDED GROQ CHAT COMPLETION
# ============================================================

def create_chat_completion(
    messages: list[dict[str, str]],
    budget: LLMCallBudget,
    temperature: float = 0,
    max_completion_tokens: int | None = None,
):
    """
    Create one guarded Groq chat completion.

    Every Groq API call consumes one call from the
    request-level LLM budget.
    """

    budget.consume()

    request_kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    if max_completion_tokens is not None:
        request_kwargs["max_completion_tokens"] = (
            max_completion_tokens
        )

    return _client.chat.completions.create(
        **request_kwargs
    )


# ============================================================
# EXPLANATION GENERATION
# ============================================================

def generate_explanation(
    system_prompt: str,
    user_prompt: str,
    budget: LLMCallBudget | None = None,
) -> str:
    """
    Generate one explanation using the configured Groq model.

    When no budget is supplied, a fresh request-level budget
    is created for backward compatibility.
    """

    if budget is None:
        budget = LLMCallBudget()

    completion = create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        budget=budget,
        temperature=0.2,
        max_completion_tokens=500,
    )

    explanation = (
        completion
        .choices[0]
        .message
        .content
    )

    if not explanation:
        raise RuntimeError(
            "Groq returned an empty explanation."
        )

    return explanation.strip()