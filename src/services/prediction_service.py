"""
AlphaPulse prediction application service.

Provides the application-level interface for generating
machine-learning trend predictions.

The service layer intentionally sits between delivery layers
such as FastAPI/serverless handlers and the ML inference layer.
"""

from __future__ import annotations

from src.ml.inference.predictor import predict_trend


SUPPORTED_ASSETS = {
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "ADA",
    "BITCOIN",
    "ETHEREUM",
    "SOLANA",
    "RIPPLE",
    "CARDANO",
}

SUPPORTED_HORIZONS = {
    6,
    12,
    24,
}


def validate_asset(
    asset: str,
) -> str:
    """
    Normalize and validate an asset identifier.
    """

    if not isinstance(asset, str):
        raise ValueError(
            "Asset must be a cryptocurrency symbol or name."
        )

    normalized_asset = asset.strip().upper()

    if not normalized_asset:
        raise ValueError(
            "Asset must not be empty."
        )

    if normalized_asset not in SUPPORTED_ASSETS:
        raise ValueError(
            f"Unsupported asset '{asset}'. "
            "Supported assets are BTC, ETH, SOL, XRP and ADA."
        )

    return normalized_asset


def validate_horizon(
    horizon_hours: int,
) -> int:
    """
    Normalize and validate a prediction horizon.
    """

    try:
        normalized_horizon = int(
            horizon_hours
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Prediction horizon must be "
            "6, 12, or 24 hours."
        ) from error

    if normalized_horizon not in SUPPORTED_HORIZONS:
        raise ValueError(
            f"Unsupported prediction horizon: "
            f"{normalized_horizon}. "
            "Supported horizons are 6, 12, and 24."
        )

    return normalized_horizon


def generate_prediction(
    asset: str,
    horizon_hours: int,
) -> dict:
    """
    Generate one AlphaPulse production prediction.

    This is the application-level prediction use case.
    """

    normalized_asset = validate_asset(
        asset
    )

    normalized_horizon = validate_horizon(
        horizon_hours
    )

    return predict_trend(
        asset=normalized_asset,
        horizon_hours=normalized_horizon,
    )