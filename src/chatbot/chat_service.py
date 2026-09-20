"""
Main orchestration service for the AlphaPulse chatbot.

This module connects:

    natural-language prompt
        ->
    Groq intent parsing
        ->
    deterministic execution planning
        ->
    AlphaPulse model execution
        ->
    grounded response generation

The LLM interprets user language.
It does NOT generate prices, probabilities, predictions,
returns, or investment calculations.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.chatbot.intent_parser import (
    parse_user_intent,
)
from src.chatbot.planner import (
    create_execution_plan,
)
from src.chatbot.executor import (
    execute_plan,
)
from src.chatbot.response_builder import (
    build_chat_response,
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_MESSAGE_LENGTH = 2000


# ============================================================
# MESSAGE VALIDATION
# ============================================================

def validate_message(
    message: str,
) -> str:
    """
    Validate a raw chatbot message before sending anything
    to the LLM or prediction pipeline.
    """

    if not isinstance(message, str):
        raise ValueError(
            "Chat message must be text."
        )

    cleaned = message.strip()

    if not cleaned:
        raise ValueError(
            "Chat message cannot be empty."
        )

    if len(cleaned) > MAX_MESSAGE_LENGTH:
        raise ValueError(
            f"Chat message cannot exceed "
            f"{MAX_MESSAGE_LENGTH} characters."
        )

    return cleaned


# ============================================================
# MAIN CHAT SERVICE
# ============================================================

def process_chat_message(
    message: str,
) -> dict[str, Any]:
    """
    Process one AlphaPulse chatbot message end-to-end.

    Returns a JSON-safe dictionary that can later be returned
    directly by the FastAPI POST /api/chat endpoint.
    """

    cleaned_message = validate_message(
        message
    )

    # --------------------------------------------------------
    # 1. Natural language -> structured intent
    # --------------------------------------------------------

    intent = parse_user_intent(
        cleaned_message
    )

    # --------------------------------------------------------
    # 2. Structured intent -> deterministic execution plan
    # --------------------------------------------------------

    plan = create_execution_plan(
        intent
    )

    # --------------------------------------------------------
    # 3. Execute approved AlphaPulse operations
    # --------------------------------------------------------

    result = execute_plan(
        plan
    )

    # --------------------------------------------------------
    # 4. Structured result -> grounded human-readable answer
    # --------------------------------------------------------

    response_text = build_chat_response(
        result
    )

    # --------------------------------------------------------
    # 5. Build API-safe response
    # --------------------------------------------------------

    return {
        "success": result.success,

        "message": response_text,

        "intent": {
            "type": intent.intent_type.value,
            "coins": intent.coins,
            "horizon_hours": intent.horizon_hours,
            "investment_amount": intent.investment_amount,
            "currency": intent.currency,
            "requested_horizon_text":
                intent.requested_horizon_text,
            "is_supported": intent.is_supported,
            "unsupported_reason":
                intent.unsupported_reason,
        },

        "predictions": result.predictions,

        "market_data": result.market_data,
    }


# ============================================================
# OPTIONAL DEBUG VERSION
# ============================================================

def process_chat_message_debug(
    message: str,
) -> dict[str, Any]:
    """
    Debug helper exposing the complete internal structures.

    Do not use this response format for the public API.
    """

    cleaned_message = validate_message(
        message
    )

    intent = parse_user_intent(
        cleaned_message
    )

    plan = create_execution_plan(
        intent
    )

    result = execute_plan(
        plan
    )

    response_text = build_chat_response(
        result
    )

    return {
        "message": response_text,
        "intent": asdict(intent),
        "plan": asdict(plan),
        "result": asdict(result),
    }
