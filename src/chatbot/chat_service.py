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
        ->
    persistent PostgreSQL chat memory

The LLM interprets user language.
It does NOT generate prices, probabilities, predictions,
returns, or investment calculations.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from src.chatbot.intent_parser import parse_user_intent
from src.chatbot.planner import create_execution_plan
from src.chatbot.executor import execute_plan
from src.chatbot.response_builder import build_chat_response

from src.repositories.chat_memory_repository import (
    get_or_create_session,
    get_recent_messages,
    save_chat_message,
)

from src.infrastructure.llm.groq_client import (
    LLMCallBudget,
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_MESSAGE_LENGTH = 2000
MAX_MEMORY_MESSAGES = 10


# ============================================================
# MESSAGE VALIDATION
# ============================================================

def validate_message(message: str) -> str:
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
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Process one AlphaPulse chatbot message end-to-end.

    If a valid session_id is supplied, the existing conversation
    is reused. Otherwise a new persistent chat session is created.

    Recent conversation history is loaded from PostgreSQL and
    supplied to the intent parser so follow-up messages can reuse
    relevant context.

    A fresh LLM call budget is created for every user message.

    Both the user message and assistant response are stored in
    PostgreSQL after processing.
    """

    cleaned_message = validate_message(
        message
    )

    # --------------------------------------------------------
    # Create request-level LLM call budget
    # --------------------------------------------------------

    llm_budget = LLMCallBudget()

    # --------------------------------------------------------
    # 1. Get or create persistent conversation session
    # --------------------------------------------------------

    active_session_id = get_or_create_session(
        session_id
    )

    # --------------------------------------------------------
    # 2. Load recent conversation context
    # --------------------------------------------------------

    conversation_history = get_recent_messages(
        active_session_id,
        limit=MAX_MEMORY_MESSAGES,
    )

    # --------------------------------------------------------
    # 3. Natural language -> structured intent
    # --------------------------------------------------------

    intent = parse_user_intent(
        cleaned_message,
        conversation_history=conversation_history,
        llm_budget=llm_budget,
    )

    # --------------------------------------------------------
    # 4. Structured intent -> deterministic execution plan
    # --------------------------------------------------------

    plan = create_execution_plan(
        intent
    )

    # --------------------------------------------------------
    # 5. Execute approved AlphaPulse operations
    # --------------------------------------------------------

    result = execute_plan(
        plan
    )

    # --------------------------------------------------------
    # 6. Structured result -> grounded human-readable answer
    # --------------------------------------------------------

    response_text = build_chat_response(
        result
    )

    # --------------------------------------------------------
    # 7. Persist conversation memory
    # --------------------------------------------------------

    intent_type = intent.intent_type.value

    save_chat_message(
        session_id=active_session_id,
        role="user",
        content=cleaned_message,
        intent_type=intent_type,
        coins=intent.coins,
        horizon_hours=intent.horizon_hours,
    )

    save_chat_message(
        session_id=active_session_id,
        role="assistant",
        content=response_text,
        intent_type=intent_type,
        coins=intent.coins,
        horizon_hours=intent.horizon_hours,
    )

    # --------------------------------------------------------
    # 8. Build API-safe response
    # --------------------------------------------------------

    return {
        "success": result.success,

        "session_id": active_session_id,

        "message": response_text,

        "intent": {
            "type": intent_type,
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

    This helper intentionally does not persist or load chat memory.
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