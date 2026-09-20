"""
Structured schemas used by the AlphaPulse chatbot.

The LLM converts natural-language questions into ChatIntent objects.
The application validates and executes those intents using the real
AlphaPulse prediction system.

The LLM is never treated as the source of prediction values,
market prices, returns, or investment calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# SUPPORTED ALPHAPULSE CAPABILITIES
# ============================================================

SUPPORTED_COINS = {
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "ADA",
}

SUPPORTED_HORIZONS = {
    6,
    12,
    24,
}


# ============================================================
# INTENT TYPES
# ============================================================

class IntentType(str, Enum):
    """
    High-level actions understood by the AlphaPulse assistant.
    """

    PREDICTION = "prediction"
    INVESTMENT_PROJECTION = "investment_projection"
    COMPARISON = "comparison"
    MARKET_ANALYSIS = "market_analysis"
    INVESTMENT_SUGGESTION = "investment_suggestion"
    CURRENT_PRICE = "current_price"
    CAPABILITY = "capability"
    IRRELEVANT = "irrelevant"


# ============================================================
# REQUEST SCHEMA
# ============================================================

@dataclass
class ChatIntent:
    """
    Structured representation of a user's natural-language request.
    """

    intent_type: IntentType

    coins: list[str] = field(default_factory=list)

    horizon_hours: Optional[int] = None

    investment_amount: Optional[float] = None

    currency: str = "USD"

    original_prompt: str = ""

    requested_horizon_text: Optional[str] = None

    requires_prediction: bool = False

    is_supported: bool = True

    unsupported_reason: Optional[str] = None


# ============================================================
# PREDICTION CALL
# ============================================================

@dataclass
class PredictionCall:
    """
    A single validated AlphaPulse model call.

    Only supported assets and supported horizons may reach
    the prediction executor.
    """

    coin: str
    horizon_hours: int


# ============================================================
# EXECUTION PLAN
# ============================================================

@dataclass
class ExecutionPlan:
    """
    Deterministic execution plan generated after intent parsing.

    The LLM does not decide which Python functions are executed.
    """

    intent: ChatIntent

    prediction_calls: list[PredictionCall] = field(
        default_factory=list
    )

    needs_market_data: bool = False

    investment_context: bool = False


# ============================================================
# CHAT RESPONSE
# ============================================================

@dataclass
class ChatResult:
    """
    Final structured result before natural-language rendering.

    Investment amount may be included as context, but AlphaPulse
    must not calculate a future dollar value unless a genuine
    numeric return/price forecast is available.
    """

    success: bool

    intent: ChatIntent

    predictions: list[dict] = field(
        default_factory=list
    )

    market_data: list[dict] = field(
        default_factory=list
    )

    message: Optional[str] = None

    error: Optional[str] = None
