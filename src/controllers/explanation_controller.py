"""
AlphaPulse explanation controller.

Provides the delivery-layer interface for explanation requests.

HTTP entry points should delegate explanation use cases to this
controller instead of calling the application service directly.
"""

from __future__ import annotations

from src.services.explanation_service import (
    explain_prediction,
)


def get_prediction_explanation(
    asset: str,
    horizon_hours: int,
    question: str | None = None,
) -> dict:
    """
    Handle an explanation request.

    Prediction generation, grounded context construction,
    and LLM explanation are delegated to the explanation
    application service.
    """

    return explain_prediction(
        asset=asset,
        horizon_hours=horizon_hours,
        question=question,
    )