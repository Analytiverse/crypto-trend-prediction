"""
AlphaPulse Groq infrastructure client.

Owns the technical integration with the Groq API.
Application services should use this module instead of
creating Groq clients directly.
"""

from __future__ import annotations

import os

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

_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# GROQ CLIENT
# ============================================================

def generate_explanation(
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Generate one explanation using the configured Groq model.
    """

    completion = _client.chat.completions.create(
        model=LLM_MODEL,
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