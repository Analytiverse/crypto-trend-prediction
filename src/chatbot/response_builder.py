"""
Grounded response builder for the AlphaPulse chatbot.

This layer converts structured AlphaPulse results into
human-readable answers.

Important:
- Prediction values come only from AlphaPulse model output.
- Market prices come only from verified AlphaPulse data.
- Probabilities are never interpreted as percentage returns.
- No future profit is invented.
"""

from __future__ import annotations

from typing import Any

from src.chatbot.schemas import (
    ChatResult,
    IntentType,
)


# ============================================================
# FORMATTING HELPERS
# ============================================================

def format_price(value: Any) -> str:
    """
    Format a USD cryptocurrency price safely.
    """

    try:
        price = float(value)
    except (TypeError, ValueError):
        return "unavailable"

    if price >= 1000:
        return f"${price:,.2f}"

    if price >= 1:
        return f"${price:,.4f}"

    return f"${price:,.6f}"


def format_percent_probability(value: Any) -> str:
    """
    Convert a probability such as 0.537619 to 53.76%.

    These are classification probabilities, NOT predicted returns.
    """

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return "unavailable"

    return f"{probability * 100:.2f}%"


def format_market_percent(value: Any) -> str:
    """
    Format an already-percent market value.

    Example:
        -0.87264 -> -0.87%

    Unlike model probabilities, this value is already expressed
    as a percentage by the market-data pipeline.
    """

    try:
        percentage = float(value)
    except (TypeError, ValueError):
        return "unavailable"

    return f"{percentage:.2f}%"


def format_confidence(prediction: dict) -> str:
    """
    Prefer the already-calculated confidence percentage returned
    by AlphaPulse.
    """

    value = prediction.get(
        "model_confidence_percent"
    )

    if value is not None:
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            pass

    value = prediction.get(
        "model_confidence"
    )

    return format_percent_probability(
        value
    )


# ============================================================
# MARKET DATA TEXT
# ============================================================

def build_market_price_summary(
    market_data: list[dict],
) -> str:
    """
    Build a grounded current-price response from verified
    PostgreSQL market snapshot data.

    The word "latest stored" is intentional because this data
    comes from AlphaPulse's database and should not be presented
    as an invented real-time exchange quote.
    """

    if not market_data:
        return (
            "No verified AlphaPulse market data is currently "
            "available for the requested cryptocurrency."
        )

    sections: list[str] = []

    for market in market_data:

        symbol = market.get(
            "symbol",
            "Unknown",
        )

        price = format_price(
            market.get("current_price")
        )

        timestamp = market.get(
            "timestamp",
            "unavailable",
        )

        change_24h = format_market_percent(
            market.get(
                "price_change_percentage_24h"
            )
        )

        high_24h = format_price(
            market.get("high_24h")
        )

        low_24h = format_price(
            market.get("low_24h")
        )

        section = (
            f"The latest stored {symbol} price in AlphaPulse is "
            f"{price}, recorded at {timestamp}. "
            f"The stored 24-hour change is {change_24h}, "
            f"with a 24-hour high of {high_24h} and "
            f"a 24-hour low of {low_24h}."
        )

        sections.append(section)

    return " ".join(sections)


# ============================================================
# PREDICTION TEXT
# ============================================================

def build_prediction_summary(
    prediction: dict,
) -> str:
    """
    Build a grounded summary for one prediction.
    """

    asset = prediction.get(
        "asset",
        "Unknown asset",
    )

    horizon = prediction.get(
        "prediction_horizon_hours"
    )

    trend = prediction.get(
        "predicted_trend",
        "UNKNOWN",
    )

    current_price = format_price(
        prediction.get(
            "current_price"
        )
    )

    confidence = format_confidence(
        prediction
    )

    probabilities = prediction.get(
        "probabilities",
        {},
    )

    down = format_percent_probability(
        probabilities.get("DOWN")
    )

    stable = format_percent_probability(
        probabilities.get("STABLE")
    )

    up = format_percent_probability(
        probabilities.get("UP")
    )

    timestamp = prediction.get(
        "prediction_timestamp",
        "unavailable",
    )

    return (
        f"AlphaPulse predicts {asset} to be {trend} "
        f"over the next {horizon} hours. "
        f"The model confidence is {confidence}. "
        f"Class probabilities are DOWN {down}, "
        f"STABLE {stable}, and UP {up}. "
        f"The latest price used by the prediction pipeline is "
        f"{current_price}, with data timestamp {timestamp}."
    )


# ============================================================
# COMPARISON
# ============================================================

def build_comparison_summary(
    predictions: list[dict],
) -> str:
    """
    Compare multiple real AlphaPulse prediction outputs.

    Comparison is deterministic. We do not ask the LLM
    to decide which probabilities are larger.
    """

    if not predictions:
        return (
            "No prediction results are available "
            "for comparison."
        )

    sections: list[str] = []

    for prediction in predictions:

        asset = prediction.get(
            "asset",
            "Unknown",
        )

        trend = prediction.get(
            "predicted_trend",
            "UNKNOWN",
        )

        confidence = format_confidence(
            prediction
        )

        probabilities = prediction.get(
            "probabilities",
            {},
        )

        up = format_percent_probability(
            probabilities.get("UP")
        )

        down = format_percent_probability(
            probabilities.get("DOWN")
        )

        stable = format_percent_probability(
            probabilities.get("STABLE")
        )

        sections.append(
            f"{asset}: {trend} "
            f"(confidence {confidence}; "
            f"UP {up}, STABLE {stable}, DOWN {down})"
        )

    # --------------------------------------------------------
    # Find strongest UP probability
    # --------------------------------------------------------

    valid_up_predictions = []

    for prediction in predictions:

        probabilities = prediction.get(
            "probabilities",
            {},
        )

        up_probability = probabilities.get(
            "UP"
        )

        if isinstance(
            up_probability,
            (int, float),
        ):
            valid_up_predictions.append(
                prediction
            )

    comparison_note = ""

    if valid_up_predictions:

        strongest_up = max(
            valid_up_predictions,
            key=lambda item: (
                item.get(
                    "probabilities",
                    {},
                ).get(
                    "UP",
                    -1,
                )
            ),
        )

        strongest_asset = strongest_up.get(
            "asset",
            "Unknown",
        )

        strongest_probability = (
            strongest_up
            .get("probabilities", {})
            .get("UP")
        )

        strongest_probability_text = (
            format_percent_probability(
                strongest_probability
            )
        )

        comparison_note = (
            f" Among these model outputs, "
            f"{strongest_asset} has the highest UP "
            f"probability at "
            f"{strongest_probability_text}."
        )

    return (
        " | ".join(sections)
        + comparison_note
    )


# ============================================================
# INVESTMENT SCENARIO
# ============================================================

def build_investment_summary(
    result: ChatResult,
) -> str:
    """
    Explain an investment scenario without inventing
    a percentage return or future portfolio value.
    """

    intent = result.intent

    if not result.predictions:
        return (
            result.message
            or
            "No AlphaPulse prediction is available "
            "for this investment scenario."
        )

    prediction = result.predictions[0]

    asset = prediction.get(
        "asset",
        "the selected asset",
    )

    trend = prediction.get(
        "predicted_trend",
        "UNKNOWN",
    )

    confidence = format_confidence(
        prediction
    )

    probabilities = prediction.get(
        "probabilities",
        {},
    )

    up = format_percent_probability(
        probabilities.get("UP")
    )

    stable = format_percent_probability(
        probabilities.get("STABLE")
    )

    down = format_percent_probability(
        probabilities.get("DOWN")
    )

    amount = intent.investment_amount

    if amount is not None:
        amount_text = (
            f"{intent.currency} "
            f"{amount:,.2f}"
        )
    else:
        amount_text = (
            "the provided investment amount"
        )

    horizon = prediction.get(
        "prediction_horizon_hours"
    )

    # Prefer the verified market snapshot price when available.
    if result.market_data:
        current_price = format_price(
            result.market_data[0].get(
                "current_price"
            )
        )

        price_timestamp = result.market_data[0].get(
            "timestamp",
            "unavailable",
        )

        price_text = (
            f"The latest stored market price is "
            f"{current_price}, recorded at "
            f"{price_timestamp}. "
        )

    else:
        current_price = format_price(
            prediction.get(
                "current_price"
            )
        )

        price_text = (
            f"The latest price used by the prediction pipeline "
            f"is {current_price}. "
        )

    return (
        f"For your {amount_text} {asset} scenario, "
        f"AlphaPulse predicts {asset} to be {trend} "
        f"over the next {horizon} hours with "
        f"{confidence} model confidence. "
        f"The model probabilities are UP {up}, "
        f"STABLE {stable}, and DOWN {down}. "
        f"{price_text}"
        f"The current AlphaPulse model predicts market direction, "
        f"not an exact percentage return or future price. "
        f"Therefore these probabilities cannot be converted into "
        f"an exact profit, loss, or future portfolio value."
    )


# ============================================================
# MAIN RESPONSE BUILDER
# ============================================================

def build_chat_response(
    result: ChatResult,
) -> str:
    """
    Convert a ChatResult into the final grounded chatbot response.
    """

    # --------------------------------------------------------
    # Errors / unsupported requests
    # --------------------------------------------------------

    if not result.success:

        if result.error:
            return (
                "AlphaPulse could not complete the request. "
                f"{result.error}"
            )

        if result.message:
            return result.message

        return (
            "AlphaPulse could not complete this request."
        )

    # --------------------------------------------------------
    # Executor already supplied a complete message
    # --------------------------------------------------------

    if (
        result.message
        and result.intent.intent_type
        in {
            IntentType.IRRELEVANT,
            IntentType.CAPABILITY,
        }
    ):
        return result.message

    # --------------------------------------------------------
    # Current verified market price
    # --------------------------------------------------------

    if (
        result.intent.intent_type
        == IntentType.CURRENT_PRICE
    ):
        return build_market_price_summary(
            result.market_data
        )

    # --------------------------------------------------------
    # Investment scenario
    # --------------------------------------------------------

    if (
        result.intent.intent_type
        == IntentType.INVESTMENT_PROJECTION
    ):
        return build_investment_summary(
            result
        )
         # --------------------------------------------------------
         # Combined market data + prediction
         # --------------------------------------------------------

    if (
        result.market_data
        and result.predictions
    ):
        market_text = build_market_price_summary(
            result.market_data
        )

        if len(result.predictions) > 1:
            prediction_text = build_comparison_summary(
                result.predictions
            )
        else:
            prediction_text = build_prediction_summary(
                result.predictions[0]
            )

        return (
            f"{market_text} "
            f"{prediction_text}"
        )
    # --------------------------------------------------------
    # Comparison / investment suggestion
    # --------------------------------------------------------

    if (
        result.intent.intent_type
        in {
            IntentType.COMPARISON,
            IntentType.INVESTMENT_SUGGESTION,
        }
        and len(result.predictions) > 1
    ):
        return build_comparison_summary(
            result.predictions
        )

    # --------------------------------------------------------
    # Single prediction
    # --------------------------------------------------------

    if len(result.predictions) == 1:
        return build_prediction_summary(
            result.predictions[0]
        )

    # --------------------------------------------------------
    # Multiple predictions
    # --------------------------------------------------------

    if len(result.predictions) > 1:
        return build_comparison_summary(
            result.predictions
        )

    # --------------------------------------------------------
    # Existing message
    # --------------------------------------------------------

    if result.message:
        return result.message

    return (
        "I don't have enough verified AlphaPulse data "
        "to answer that request."
    )