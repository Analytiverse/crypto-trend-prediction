"""
AlphaPulse explanation application service.

The machine-learning system generates the market prediction.
The Groq-hosted LLM only explains that prediction using
controlled AlphaPulse prediction context.
"""

from __future__ import annotations

from src.infrastructure.llm.groq_client import (
    LLM_MODEL,
    generate_explanation,
)
from src.services.prediction_service import (
    generate_prediction,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the explanation assistant for AlphaPulse,
a cryptocurrency trend-analysis system.

AlphaPulse uses a trained machine-learning model to classify
short-term cryptocurrency movement as:

- UP
- DOWN
- STABLE

Your job is ONLY to explain the prediction supplied to you.

Important rules:

1. Never change or override the ML model prediction.

2. Never claim that you generated the market prediction.

3. Use only the AlphaPulse prediction data supplied in the prompt.

4. Never invent:
   - cryptocurrency news
   - market events
   - technical indicators
   - feature values
   - prices
   - probabilities
   - reasons or causes not supplied in the context

5. Clearly communicate uncertainty.

6. A confidence value such as 40% represents substantial
   uncertainty and must not be described as a strong prediction.

7. If class probabilities are close together, explicitly mention
   that the model does not have a strongly separated signal.

8. Do not recommend buying, selling, holding, or shorting an asset.

9. Do not present predictions as guaranteed future outcomes.

10. Keep the explanation concise and understandable to a
    non-technical user.

11. Explain that the result is an analytical model signal and
    not financial advice.
""".strip()


# ============================================================
# BUILD GROUNDED CONTEXT
# ============================================================

def build_prediction_context(
    prediction: dict,
) -> str:
    """
    Convert an AlphaPulse prediction into controlled LLM context.
    """

    probabilities = prediction[
        "probabilities"
    ]

    threshold_percent = (
        prediction["threshold"] * 100
    )

    return f"""
ALPHAPULSE PREDICTION

Asset: {prediction["asset"]}

Current price:
${prediction["current_price"]:,.6f}

Prediction horizon:
{prediction["prediction_horizon_hours"]} hours

Prediction timestamp:
{prediction["prediction_timestamp"]}

Predicted trend:
{prediction["predicted_trend"]}

Model confidence:
{prediction["model_confidence_percent"]:.2f}%

Class probabilities:
DOWN: {probabilities["DOWN"] * 100:.2f}%
STABLE: {probabilities["STABLE"] * 100:.2f}%
UP: {probabilities["UP"] * 100:.2f}%

Movement threshold:
±{threshold_percent:.2f}%

Prediction model:
{prediction["model"]}

Model version:
{prediction["model_version"]}

Feature count:
{prediction["feature_count"]}
""".strip()


# ============================================================
# EXPLANATION SERVICE
# ============================================================

def explain_prediction(
    asset: str,
    horizon_hours: int,
    question: str | None = None,
) -> dict:
    """
    Generate a grounded natural-language explanation for an
    AlphaPulse machine-learning prediction.
    """

    # --------------------------------------------------------
    # 1. Generate prediction through the application service
    # --------------------------------------------------------

    prediction = generate_prediction(
        asset=asset,
        horizon_hours=horizon_hours,
    )

    # --------------------------------------------------------
    # 2. Convert prediction to controlled LLM context
    # --------------------------------------------------------

    prediction_context = (
        build_prediction_context(
            prediction
        )
    )

    # --------------------------------------------------------
    # 3. Default user question
    # --------------------------------------------------------

    if not question:
        question = (
            "Explain this prediction in simple language. "
            "Explain the predicted direction, confidence, "
            "the competing class probabilities, and how "
            "certain or uncertain the model currently is."
        )

    user_prompt = f"""
{prediction_context}

USER QUESTION:
{question}

Answer using ONLY the AlphaPulse prediction information above.
Do not introduce external market information.
""".strip()

    # --------------------------------------------------------
    # 4. Call Groq infrastructure
    # --------------------------------------------------------

    explanation = generate_explanation(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    # --------------------------------------------------------
    # 5. Structured response
    # --------------------------------------------------------

    return {
        "asset":
            prediction["asset"],
        "horizon_hours":
            prediction["prediction_horizon_hours"],
        "prediction_timestamp":
            prediction["prediction_timestamp"],
        "predicted_trend":
            prediction["predicted_trend"],
        "model_confidence":
            prediction["model_confidence"],
        "probabilities":
            prediction["probabilities"],
        "ml_model":
            prediction["model"],
        "llm_provider":
            "Groq",
        "llm_model":
            LLM_MODEL,
        "question":
            question,
        "explanation":
            explanation,
        "disclaimer":
            (
                "AlphaPulse provides model-generated "
                "analytical signals and explanations. "
                "It is not financial advice."
            ),
    }