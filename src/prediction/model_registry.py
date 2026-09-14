"""
AlphaPulse model registry.

Responsible for mapping prediction horizons to the correct
model and metadata artifacts.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "metadata"
)


SUPPORTED_HORIZONS = [
    6,
    12,
    24,
]


MODEL_FILES = {
    6:
        MODEL_DIR
        / "model_6h.joblib",

    12:
        MODEL_DIR
        / "model_12h.joblib",

    24:
        MODEL_DIR
        / "model_24h.joblib",
}


METADATA_FILES = {
    6:
        METADATA_DIR
        / "model_6h.json",

    12:
        METADATA_DIR
        / "model_12h.json",

    24:
        METADATA_DIR
        / "model_24h.json",
}


def validate_horizon(
    horizon_hours: int,
) -> int:
    """
    Validate requested prediction horizon.
    """

    try:
        horizon_hours = int(
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

    if horizon_hours not in (
        SUPPORTED_HORIZONS
    ):
        raise ValueError(
            "Unsupported prediction horizon: "
            f"{horizon_hours}. "
            "Supported horizons are 6, 12, and 24."
        )

    return horizon_hours


@lru_cache(
    maxsize=3
)
def load_model(
    horizon_hours: int,
):
    """
    Load and cache the correct model for a horizon.
    """

    horizon_hours = (
        validate_horizon(
            horizon_hours
        )
    )

    model_path = (
        MODEL_FILES[
            horizon_hours
        ]
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model artifact not found: "
            f"{model_path}\n"
            "Run:\n"
            "python -m "
            "src.modeling.train_production_models"
        )

    return joblib.load(
        model_path
    )


@lru_cache(
    maxsize=3
)
def load_metadata(
    horizon_hours: int,
) -> dict:
    """
    Load metadata for the requested horizon.
    """

    horizon_hours = (
        validate_horizon(
            horizon_hours
        )
    )

    metadata_path = (
        METADATA_FILES[
            horizon_hours
        ]
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Metadata artifact not found: "
            f"{metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )