"""PostgreSQL repository for persistent chatbot conversation memory."""

from __future__ import annotations

from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import text

from src.infrastructure.database.connection import engine


def create_chat_session() -> str:
    """Create a new chat session and return its UUID."""

    session_id = uuid4()

    query = text(
        """
        INSERT INTO chat_sessions (id)
        VALUES (:session_id)
        """
    )

    with engine.begin() as connection:
        connection.execute(
            query,
            {"session_id": session_id},
        )

    return str(session_id)


def session_exists(session_id: str) -> bool:
    """Return True when the supplied chat session exists."""

    try:
        parsed_session_id = UUID(session_id)
    except (ValueError, TypeError, AttributeError):
        return False

    query = text(
        """
        SELECT 1
        FROM chat_sessions
        WHERE id = :session_id
        LIMIT 1
        """
    )

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"session_id": parsed_session_id},
        ).scalar()

    return result is not None


def save_chat_message(
    session_id: str,
    role: str,
    content: str,
    intent_type: Optional[str] = None,
    coins: Optional[list[str]] = None,
    horizon_hours: Optional[int] = None,
) -> None:
    """Persist one user or assistant message."""

    if role not in {"user", "assistant"}:
        raise ValueError("role must be either 'user' or 'assistant'")

    parsed_session_id = UUID(session_id)

    query = text(
        """
        INSERT INTO chat_messages (
            session_id,
            role,
            content,
            intent_type,
            coins,
            horizon_hours
        )
        VALUES (
            :session_id,
            :role,
            :content,
            :intent_type,
            :coins,
            :horizon_hours
        )
        """
    )

    update_session_query = text(
        """
        UPDATE chat_sessions
        SET updated_at = NOW()
        WHERE id = :session_id
        """
    )

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "session_id": parsed_session_id,
                "role": role,
                "content": content,
                "intent_type": intent_type,
                "coins": coins,
                "horizon_hours": horizon_hours,
            },
        )

        connection.execute(
            update_session_query,
            {"session_id": parsed_session_id},
        )


def get_recent_messages(
    session_id: str,
    limit: int = 10,
) -> list[dict]:
    """
    Return the most recent messages in chronological order.

    The SQL query retrieves newest messages first so LIMIT applies to the
    latest conversation context. The result is then reversed before returning.
    """

    parsed_session_id = UUID(session_id)

    query = text(
        """
        SELECT
            role,
            content,
            intent_type,
            coins,
            horizon_hours,
            created_at
        FROM chat_messages
        WHERE session_id = :session_id
        ORDER BY created_at DESC, id DESC
        LIMIT :limit
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(
            query,
            {
                "session_id": parsed_session_id,
                "limit": limit,
            },
        ).mappings().all()

    messages = [dict(row) for row in rows]
    messages.reverse()

    return messages


def get_or_create_session(session_id: Optional[str] = None) -> str:
    """
    Reuse a valid existing session or create a new one.

    Invalid or unknown session IDs do not cause the chatbot request to fail.
    """

    if session_id and session_exists(session_id):
        return session_id

    return create_chat_session()