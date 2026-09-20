"""
Deterministic execution planner for the AlphaPulse chatbot.

The intent parser understands natural language.
The planner decides which AlphaPulse operations are allowed.

Important security rule:

    LLM -> structured intent -> deterministic planner -> model

The LLM never supplies Python function names and never executes
arbitrary operations.
"""

from __future__ import annotations

from src.chatbot.schemas import (
    ChatIntent,
    ExecutionPlan,
    IntentType,
    PredictionCall,
    SUPPORTED_COINS,
    SUPPORTED_HORIZONS,
)


MAX_PREDICTION_CALLS = 5


# ============================================================
# VALIDATION
# ============================================================

def validate_coin(coin: str) -> str:
    """
    Validate and normalize an AlphaPulse cryptocurrency symbol.
    """

    normalized = str(coin).strip().upper()

    if normalized not in SUPPORTED_COINS:
        raise ValueError(
            f"Unsupported cryptocurrency: {normalized}. "
            "AlphaPulse currently supports BTC, ETH, SOL, XRP and ADA."
        )

    return normalized


def validate_horizon(horizon_hours: int | None) -> int:
    """
    Validate a prediction horizon before creating a model call.
    """

    if horizon_hours is None:
        raise ValueError(
            "A prediction horizon is required. "
            "AlphaPulse currently supports 6, 12 and 24 hours."
        )

    try:
        horizon = int(horizon_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Prediction horizon must be a valid number of hours."
        ) from exc

    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(
            f"Unsupported prediction horizon: {horizon} hours. "
            "AlphaPulse currently supports 6, 12 and 24 hours."
        )

    return horizon


# ============================================================
# PROMPT REQUIREMENT HELPERS
# ============================================================

def prompt_requests_market_data(
    intent: ChatIntent,
) -> bool:
    """
    Detect whether the original user prompt explicitly requests
    current/latest market information in addition to its main intent.

    This is intentionally deterministic.

    Examples:
        "What is the current BTC price?"
        "What is the current BTC price and prediction for 24 hours?"
        "Show me the latest ETH price."
        "Analyze BTC market cap and 24h volume."
    """

    prompt = (
        intent.original_prompt
        or ""
    ).strip().lower()

    if not prompt:
        return False

    # Explicit market-data phrases that do not depend on word order.
    market_phrases = (
        "market data",
        "market cap",
        "market capitalization",
        "trading volume",
        "total volume",
        "24h high",
        "24 hour high",
        "24-hour high",
        "24h low",
        "24 hour low",
        "24-hour low",
        "24h change",
        "24 hour change",
        "24-hour change",
    )

    if any(
        phrase in prompt
        for phrase in market_phrases
    ):
        return True

    # Handle prompts such as:
    # "current BTC price"
    # "latest ETH price"
    # "BTC price right now"
    #
    # We deliberately do not require "current price" to be
    # consecutive words because the coin name may appear between them.
    price_requested = (
        "price" in prompt
    )

    current_price_requested = (
        "current" in prompt
        or "latest" in prompt
        or "right now" in prompt
        or "price now" in prompt
    )

    if (
        price_requested
        and current_price_requested
    ):
        return True

    return False
    """
    Detect whether the original user prompt explicitly requests
    current/latest market information in addition to its main intent.

    This is intentionally deterministic.

    Example:
        "What is the current BTC price and prediction
         for the next 24 hours?"

    The parser may classify the overall request as PREDICTION,
    but the planner must still retrieve verified market data
    because the user explicitly requested the current price.
    """

    prompt = (
        intent.original_prompt
        or ""
    ).strip().lower()

    if not prompt:
        return False

    market_phrases = (
        "current price",
        "current market price",
        "latest price",
        "latest market price",
        "price right now",
        "price now",
        "market price",
        "current market data",
        "latest market data",
        "market data",
        "market cap",
        "market capitalization",
        "trading volume",
        "total volume",
        "24h high",
        "24 hour high",
        "24-hour high",
        "24h low",
        "24 hour low",
        "24-hour low",
        "24h change",
        "24 hour change",
        "24-hour change",
    )

    return any(
        phrase in prompt
        for phrase in market_phrases
    )


# ============================================================
# PREDICTION CALL BUILDER
# ============================================================

def build_prediction_calls(
    intent: ChatIntent,
) -> list[PredictionCall]:
    """
    Convert a validated ChatIntent into one or more model calls.

    Example:

        Compare BTC, ETH and SOL over 12 hours

    becomes:

        BTC -> 12h model
        ETH -> 12h model
        SOL -> 12h model
    """

    if not intent.requires_prediction:
        return []

    horizon = validate_horizon(
        intent.horizon_hours
    )

    if not intent.coins:
        raise ValueError(
            "At least one supported cryptocurrency is required "
            "for a prediction."
        )

    if len(intent.coins) > MAX_PREDICTION_CALLS:
        raise ValueError(
            f"A maximum of {MAX_PREDICTION_CALLS} prediction "
            "calls are allowed in one chatbot request."
        )

    calls: list[PredictionCall] = []

    seen_coins: set[str] = set()

    for coin in intent.coins:

        normalized_coin = validate_coin(
            coin
        )

        if normalized_coin in seen_coins:
            continue

        seen_coins.add(
            normalized_coin
        )

        calls.append(
            PredictionCall(
                coin=normalized_coin,
                horizon_hours=horizon,
            )
        )

    return calls


# ============================================================
# INTENT-SPECIFIC VALIDATION
# ============================================================

def validate_intent_requirements(
    intent: ChatIntent,
) -> None:
    """
    Apply deterministic business rules after LLM parsing.
    """

    if intent.intent_type == IntentType.IRRELEVANT:
        return

    if intent.intent_type == IntentType.CAPABILITY:
        return

    if intent.intent_type == IntentType.CURRENT_PRICE:

        if not intent.coins:
            raise ValueError(
                "Please specify which supported cryptocurrency "
                "you want the current price for."
            )

        return

    if intent.intent_type == IntentType.COMPARISON:

        if len(intent.coins) < 2:
            raise ValueError(
                "A comparison requires at least two "
                "supported cryptocurrencies."
            )

    if intent.intent_type == IntentType.INVESTMENT_PROJECTION:

        if intent.investment_amount is None:
            raise ValueError(
                "An investment scenario requires an "
                "investment amount."
            )

        if intent.investment_amount <= 0:
            raise ValueError(
                "Investment amount must be greater than zero."
            )

        if not intent.coins:
            raise ValueError(
                "Please specify which supported cryptocurrency "
                "the investment scenario should use."
            )

    if intent.requires_prediction:

        if not intent.coins:
            raise ValueError(
                "Please specify at least one supported "
                "cryptocurrency."
            )

        validate_horizon(
            intent.horizon_hours
        )


# ============================================================
# MAIN PLANNER
# ============================================================

def create_execution_plan(
    intent: ChatIntent,
) -> ExecutionPlan:
    """
    Create a safe deterministic execution plan.

    No external API, database query, or ML model is executed here.

    A single natural-language request may require multiple
    deterministic operations.

    Example:

        "What is the current BTC price and prediction
         for the next 24 hours?"

    may require:

        1. verified market-data retrieval
        2. BTC 24-hour model prediction
    """

    if not isinstance(intent, ChatIntent):
        raise TypeError(
            "create_execution_plan expects a ChatIntent object."
        )

    # --------------------------------------------------------
    # Irrelevant questions
    # --------------------------------------------------------

    if intent.intent_type == IntentType.IRRELEVANT:

        return ExecutionPlan(
            intent=intent,
            prediction_calls=[],
            needs_market_data=False,
            investment_context=False,
        )

    # --------------------------------------------------------
    # Unsupported request detected by parser
    # --------------------------------------------------------

    if not intent.is_supported:

        return ExecutionPlan(
            intent=intent,
            prediction_calls=[],
            needs_market_data=False,
            investment_context=(
                intent.intent_type
                == IntentType.INVESTMENT_PROJECTION
            ),
        )

    # --------------------------------------------------------
    # Deterministic validation
    # --------------------------------------------------------

    validate_intent_requirements(
        intent
    )

    # --------------------------------------------------------
    # Prediction calls
    # --------------------------------------------------------

    prediction_calls = (
        build_prediction_calls(intent)
        if intent.requires_prediction
        else []
    )

    # --------------------------------------------------------
    # Market-data requirements
    # --------------------------------------------------------

    intent_requires_market_data = (
        intent.intent_type
        in {
            IntentType.CURRENT_PRICE,
            IntentType.MARKET_ANALYSIS,
            IntentType.INVESTMENT_PROJECTION,
            IntentType.INVESTMENT_SUGGESTION,
        }
    )

    prompt_requires_market_data = (
        prompt_requests_market_data(
            intent
        )
    )

    needs_market_data = (
        intent_requires_market_data
        or prompt_requires_market_data
    )

    # --------------------------------------------------------
    # Investment context
    # --------------------------------------------------------

    investment_context = (
        intent.intent_type
        == IntentType.INVESTMENT_PROJECTION
    )

    return ExecutionPlan(
        intent=intent,
        prediction_calls=prediction_calls,
        needs_market_data=needs_market_data,
        investment_context=investment_context,
    )


# ============================================================
# DISPLAY UTILITY
# ============================================================

def describe_execution_plan(
    plan: ExecutionPlan,
) -> dict:
    """
    Convert an ExecutionPlan into a simple dictionary for debugging.
    """

    return {
        "intent_type": plan.intent.intent_type.value,

        "prediction_calls": [
            {
                "coin": call.coin,
                "horizon_hours": call.horizon_hours,
            }
            for call in plan.prediction_calls
        ],

        "needs_market_data":
            plan.needs_market_data,

        "investment_context":
            plan.investment_context,

        "is_supported":
            plan.intent.is_supported,

        "unsupported_reason":
            plan.intent.unsupported_reason,
    }