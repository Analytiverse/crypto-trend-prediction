"""
AlphaPulse prediction controller.

Provides the delivery-layer interface for prediction requests.

HTTP entry points should delegate prediction use cases to this
controller instead of calling application services directly.
"""

from __future__ import annotations

from src.services.prediction_service import (
    generate_prediction,
)


def get_prediction(
    asset: str,
    horizon_hours: int,
) -> dict:
    """
    Handle a prediction request.

    Request validation and prediction generation are delegated
    to the prediction application service.
    """

    return generate_prediction(
        asset=asset,
        horizon_hours=horizon_hours,
    )