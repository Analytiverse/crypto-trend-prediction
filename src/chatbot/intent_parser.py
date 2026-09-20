"""
Natural-language intent parser for the AlphaPulse chatbot.

Responsibilities:
- Determine whether a question belongs to AlphaPulse's domain.
- Extract cryptocurrency symbols.
- Extract requested prediction horizon.
- Extract investment amount and currency.
- Classify the requested operation.

Important:
The LLM only interprets the user's request here.
It does NOT generate prices, predictions, returns, or profit values.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from src.chatbot.schemas import (
    ChatIntent,
    IntentType,
    SUPPORTED_COINS,
    SUPPORTED_HORIZONS,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not configured. "
        "Add it to the environment or .env file."
    )


# ============================================================
# GROQ CONFIGURATION
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY
)

LLM_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the intent parser for AlphaPulse, a cryptocurrency
trend-prediction application.

Your ONLY job is to convert the user's message into structured JSON.

AlphaPulse supports these cryptocurrencies:

BTC
ETH
SOL
XRP
ADA

Aliases:

Bitcoin -> BTC
Ethereum -> ETH
Solana -> SOL
Ripple -> XRP
Cardano -> ADA

AlphaPulse's trained prediction models support ONLY these horizons:

6 hours
12 hours
24 hours

Do not invent predictions, prices, percentages, probabilities,
profits, losses, market statistics, or investment results.

Classify the request into exactly one of these intent types:

prediction
investment_projection
comparison
market_analysis
investment_suggestion
current_price
capability
irrelevant

Definitions:

prediction:
The user asks about the future trend or outlook of one or more
supported cryptocurrencies.

investment_projection:
The user provides or discusses an investment amount and asks what
could happen to that investment based on a prediction.

comparison:
The user asks to compare two or more supported cryptocurrencies.

market_analysis:
The user asks for analysis of supported cryptocurrency market data.

investment_suggestion:
The user asks which supported cryptocurrency appears preferable,
stronger, weaker, or more interesting based on AlphaPulse data/models.

current_price:
The user asks for the current/latest price of a supported coin.

capability:
The user asks what AlphaPulse can do, which coins it supports,
or which prediction horizons are available.

irrelevant:
The request is outside AlphaPulse's cryptocurrency prediction,
market-analysis, comparison, or investment-scenario scope.

Examples of irrelevant questions:

"How do I write a for loop in Python?"
"Who is the president of France?"
"Write me an email."
"What is the capital of Japan?"

Extract:

intent_type
coins
horizon_hours
investment_amount
currency
requested_horizon_text
requires_prediction
is_supported
unsupported_reason

Rules:

1. coins must contain only BTC, ETH, SOL, XRP, ADA.

2. Convert full cryptocurrency names to symbols.

3. horizon_hours may only contain 6, 12, or 24 when the requested
   horizon exactly matches a supported AlphaPulse horizon.

4. If the user requests another horizon, such as:
   2 days
   7 days
   10 days
   1 week
   1 month

   DO NOT convert it to 6, 12, or 24.

   Set horizon_hours to null.
   Preserve what the user requested in requested_horizon_text.
   Set is_supported to false.
   Explain the unsupported horizon in unsupported_reason.

5. Never approximate an unsupported horizon to the nearest
   supported horizon.

6. Extract investment amounts numerically.

Examples:

"$5k" -> 5000
"5K$" -> 5000
"$10,000" -> 10000
"2.5k USD" -> 2500

7. Default currency to USD unless another currency is explicitly
   stated.

8. requires_prediction should be true for prediction,
   investment_projection, comparison when future predictions are
   requested, and investment_suggestion when model signals are needed.

9. An irrelevant question must have:
   requires_prediction = false
   is_supported = false

10. Do not answer the user's question.

11. Return ONLY valid JSON.

Required JSON format:

{
    "intent_type": "prediction",
    "coins": ["BTC"],
    "horizon_hours": 24,
    "investment_amount": null,
    "currency": "USD",
    "requested_horizon_text": "24 hours",
    "requires_prediction": true,
    "is_supported": true,
    "unsupported_reason": null
}
"""


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(content: str) -> dict[str, Any]:
    """
    Extract and validate a JSON object from the LLM response.

    Groq is instructed to return only JSON, but this defensive
    parser also handles accidental Markdown code fences.
    """

    content = content.strip()

    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )

        content = re.sub(
            r"\s*```$",
            "",
            content,
        )

    try:
        parsed = json.loads(content)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "The intent parser returned invalid JSON."
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "The intent parser must return a JSON object."
        )

    return parsed


# ============================================================
# INTENT TYPE VALIDATION
# ============================================================

def parse_intent_type(value: Any) -> IntentType:
    """
    Convert the LLM intent string into the strict IntentType enum.
    """

    try:
        return IntentType(
            str(value).strip().lower()
        )

    except ValueError as exc:
        raise ValueError(
            f"Unknown chatbot intent type: {value}"
        ) from exc


# ============================================================
# COIN VALIDATION
# ============================================================

def validate_coins(value: Any) -> list[str]:
    """
    Ensure the LLM cannot inject unsupported assets into execution.
    """

    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(
            "Parsed coins must be a list."
        )

    coins: list[str] = []

    for coin in value:

        normalized = str(
            coin
        ).strip().upper()

        if normalized not in SUPPORTED_COINS:
            raise ValueError(
                f"Intent parser returned unsupported coin: "
                f"{normalized}"
            )

        if normalized not in coins:
            coins.append(
                normalized
            )

    return coins


# ============================================================
# HORIZON VALIDATION
# ============================================================

def validate_horizon(
    value: Any,
) -> int | None:
    """
    Validate a parsed AlphaPulse prediction horizon.

    Unsupported horizons must remain None rather than being
    silently approximated.
    """

    if value is None:
        return None

    try:
        horizon = int(value)

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "Parsed horizon must be an integer or null."
        ) from exc

    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(
            "Intent parser attempted to use an unsupported "
            f"prediction horizon: {horizon}"
        )

    return horizon


# ============================================================
# INVESTMENT AMOUNT VALIDATION
# ============================================================

def validate_investment_amount(
    value: Any,
) -> float | None:
    """
    Validate investment amount extracted by the LLM.
    """

    if value is None:
        return None

    try:
        amount = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "Investment amount must be numeric."
        ) from exc

    if amount <= 0:
        raise ValueError(
            "Investment amount must be greater than zero."
        )

    return amount


# ============================================================
# MAIN INTENT PARSER
# ============================================================

def parse_user_intent(
    user_prompt: str,
) -> ChatIntent:
    """
    Convert a natural-language user message into a validated
    AlphaPulse ChatIntent.
    """

    if not user_prompt:
        raise ValueError(
            "User prompt cannot be empty."
        )

    user_prompt = str(
        user_prompt
    ).strip()

    if not user_prompt:
        raise ValueError(
            "User prompt cannot be empty."
        )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:
        raise ValueError(
            "Groq returned an empty intent response."
        )

    parsed = extract_json(
        content
    )

    intent_type = parse_intent_type(
        parsed.get(
            "intent_type",
            "irrelevant",
        )
    )

    coins = validate_coins(
        parsed.get(
            "coins",
            [],
        )
    )

    horizon_hours = validate_horizon(
        parsed.get(
            "horizon_hours"
        )
    )

    investment_amount = (
        validate_investment_amount(
            parsed.get(
                "investment_amount"
            )
        )
    )

    currency = str(
        parsed.get(
            "currency",
            "USD",
        )
    ).strip().upper()

    requested_horizon_text = (
        parsed.get(
            "requested_horizon_text"
        )
    )

    if requested_horizon_text is not None:
        requested_horizon_text = str(
            requested_horizon_text
        ).strip()

    requires_prediction = bool(
        parsed.get(
            "requires_prediction",
            False,
        )
    )

    is_supported = bool(
        parsed.get(
            "is_supported",
            True,
        )
    )

    unsupported_reason = (
        parsed.get(
            "unsupported_reason"
        )
    )

    if unsupported_reason is not None:
        unsupported_reason = str(
            unsupported_reason
        ).strip()

    # --------------------------------------------------------
    # Application-level guardrails
    # --------------------------------------------------------

    if intent_type == IntentType.IRRELEVANT:
        requires_prediction = False
        is_supported = False

    if (
        requires_prediction
        and requested_horizon_text
        and horizon_hours is None
    ):
        is_supported = False

        if not unsupported_reason:
            unsupported_reason = (
                "The requested prediction horizon is not "
                "supported. AlphaPulse currently supports "
                "6, 12, and 24-hour predictions."
            )

    return ChatIntent(
        intent_type=intent_type,
        coins=coins,
        horizon_hours=horizon_hours,
        investment_amount=investment_amount,
        currency=currency,
        original_prompt=user_prompt,
        requested_horizon_text=requested_horizon_text,
        requires_prediction=requires_prediction,
        is_supported=is_supported,
        unsupported_reason=unsupported_reason,
    )


# ============================================================
# CLI TEST
# ============================================================

def main() -> None:
    """
    Simple local test utility.

    Example:

        python -m src.chatbot.intent_parser
    """

    print()
    print("=" * 60)
    print("ALPHAPULSE INTENT PARSER")
    print("=" * 60)

    while True:

        print()

        prompt = input(
            "Ask AlphaPulse (or type 'exit'): "
        ).strip()

        if prompt.lower() in {
            "exit",
            "quit",
        }:
            break

        try:

            intent = parse_user_intent(
                prompt
            )

            print()
            print(
                json.dumps(
                    {
                        "intent_type":
                            intent.intent_type.value,

                        "coins":
                            intent.coins,

                        "horizon_hours":
                            intent.horizon_hours,

                        "investment_amount":
                            intent.investment_amount,

                        "currency":
                            intent.currency,

                        "requested_horizon_text":
                            intent.requested_horizon_text,

                        "requires_prediction":
                            intent.requires_prediction,

                        "is_supported":
                            intent.is_supported,

                        "unsupported_reason":
                            intent.unsupported_reason,
                    },
                    indent=4,
                )
            )

        except Exception as exc:

            print()
            print(
                "[ERROR]",
                str(exc),
            )


if __name__ == "__main__":
    main()