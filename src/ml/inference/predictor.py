"""
Live prediction service for AlphaPulse.

Examples:

    python -m src.ml.inference.predictor bitcoin 6

    python -m src.ml.inference.predictor ethereum 12

    python -m src.ml.inference.predictor solana 24

Short symbols are also supported:

    python -m src.ml.inference.predictor BTC 6

    python -m src.ml.inference.predictor ETH 24
"""

from __future__ import annotations

import json
import sys

import numpy as np

from src.core.config import COINS

from src.ml.features.feature_builder import (
    ML_FEATURES,
    build_features,
    get_latest_feature_row,
    load_recent_market_data,
)

from src.ml.inference.model_registry import (
    load_metadata,
    load_model,
    validate_horizon,
)


# ============================================================
# ASSET CONFIGURATION
# ============================================================

ASSET_ALIASES = {
    "btc": "bitcoin",
    "bitcoin": "bitcoin",

    "eth": "ethereum",
    "ethereum": "ethereum",

    "sol": "solana",
    "solana": "solana",

    "xrp": "ripple",
    "ripple": "ripple",

    "ada": "cardano",
    "cardano": "cardano",
}


ASSET_SYMBOLS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "ripple": "XRP",
    "cardano": "ADA",
}


# ============================================================
# ASSET VALIDATION
# ============================================================

def normalize_asset(
    asset: str,
) -> str:
    """
    Convert symbols/names to the CoinGecko coin ID.
    """

    if not asset:
        raise ValueError(
            "Asset cannot be empty."
        )

    normalized = (
        str(asset)
        .strip()
        .lower()
    )

    coin_id = (
        ASSET_ALIASES.get(
            normalized
        )
    )

    if coin_id is None:
        raise ValueError(
            f"Unsupported asset: {asset}. "
            "Supported assets are BTC, ETH, "
            "SOL, XRP, and ADA."
        )

    if coin_id not in COINS:
        raise ValueError(
            f"{coin_id} is not configured "
            "in src/config.py."
        )

    return coin_id


# ============================================================
# PROBABILITY HELPER
# ============================================================

def build_probability_map(
    classes,
    probabilities,
) -> dict:
    """
    Convert model probability output to a consistent
    DOWN/STABLE/UP dictionary.
    """

    probability_map = {
        "DOWN": 0.0,
        "STABLE": 0.0,
        "UP": 0.0,
    }

    for label, probability in zip(
        classes,
        probabilities,
    ):

        probability_map[
            str(label)
        ] = float(
            probability
        )

    return probability_map


# ============================================================
# LIVE PREDICTION
# ============================================================

def predict_trend(
    asset: str,
    horizon_hours: int,
) -> dict:
    """
    Generate one live cryptocurrency trend prediction.

    The correct horizon-specific model is automatically loaded.

    Parameters
    ----------
    asset:
        BTC / bitcoin / ETH / ethereum / etc.

    horizon_hours:
        6, 12, or 24.

    Returns
    -------
    dict
        Structured AlphaPulse prediction.
    """

    coin_id = normalize_asset(
        asset
    )

    horizon_hours = (
        validate_horizon(
            horizon_hours
        )
    )

    # --------------------------------------------------------
    # Load latest historical observations
    # --------------------------------------------------------

    raw_df = (
        load_recent_market_data(
            lookback_hours=120
        )
    )

    # --------------------------------------------------------
    # Recreate exact training features
    # --------------------------------------------------------

    feature_df = (
        build_features(
            raw_df
        )
    )

    # --------------------------------------------------------
    # Get latest fully usable feature row
    # --------------------------------------------------------

    latest_row = (
        get_latest_feature_row(
            feature_df,
            coin_id,
        )
    )

    prediction_timestamp = (
        latest_row[
            "timestamp"
        ].iloc[0]
    )

    current_price = float(
        latest_row[
            "price"
        ].iloc[0]
    )

    # --------------------------------------------------------
    # Build model matrix
    # --------------------------------------------------------

    X = latest_row[
        ML_FEATURES
    ].copy()

    if X.isna().any().any():

        missing = (
            X.columns[
                X.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "Prediction feature row "
            "contains missing values: "
            f"{missing}"
        )

    if not np.isfinite(
        X.to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Prediction feature row contains "
            "non-finite numeric values."
        )

    # --------------------------------------------------------
    # Load correct horizon model
    # --------------------------------------------------------

    model = load_model(
        horizon_hours
    )

    metadata = (
        load_metadata(
            horizon_hours
        )
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_class = str(
        model.predict(
            X
        )[0]
    )

    # --------------------------------------------------------
    # Probabilities / model confidence
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X
        )[0]
    )

    classes = (
        model.classes_
    )

    probability_map = (
        build_probability_map(
            classes,
            probabilities,
        )
    )

    confidence = float(
        max(
            probability_map.values()
        )
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    symbol = (
        ASSET_SYMBOLS[
            coin_id
        ]
    )

    result = {
        "asset": symbol,
        "coin_id": coin_id,

        "prediction_timestamp":
            prediction_timestamp.isoformat(),

        "current_price":
            current_price,

        "prediction_horizon_hours":
            horizon_hours,

        "predicted_trend":
            predicted_class,

        "model_confidence":
            confidence,

        "model_confidence_percent":
            round(
                confidence * 100,
                2,
            ),

        "probabilities": {
            "DOWN":
                round(
                    probability_map[
                        "DOWN"
                    ],
                    6,
                ),

            "STABLE":
                round(
                    probability_map[
                        "STABLE"
                    ],
                    6,
                ),

            "UP":
                round(
                    probability_map[
                        "UP"
                    ],
                    6,
                ),
        },

        "model": (
            metadata[
                "model_name"
            ]
        ),

        "model_version": (
            metadata[
                "model_version"
            ]
        ),

        "threshold": (
            metadata[
                "threshold"
            ]
        ),

        "feature_count":
            len(
                metadata[
                    "features"
                ]
            ),

        "signal_type":
            (
                "Model-generated analytical signal"
            ),

        "disclaimer":
            (
                "This prediction is generated by "
                "AlphaPulse's statistical model. "
                "It is not financial advice and "
                "does not guarantee future performance."
            ),
    }

    return result


# ============================================================
# CLI DISPLAY
# ============================================================

def print_prediction(
    result: dict,
) -> None:

    print()
    print("=" * 60)
    print(
        "ALPHAPULSE TREND PREDICTION"
    )
    print("=" * 60)

    print(
        f"Asset:              "
        f"{result['asset']}"
    )

    print(
        f"Current Price:      "
        f"${result['current_price']:,.6f}"
    )

    print(
        f"Prediction Horizon: "
        f"{result['prediction_horizon_hours']} hours"
    )

    print(
        f"Predicted Trend:    "
        f"{result['predicted_trend']}"
    )

    print(
        f"Model Confidence:   "
        f"{result['model_confidence_percent']:.2f}%"
    )

    print(
        f"Data Timestamp:     "
        f"{result['prediction_timestamp']}"
    )

    print(
        f"Model:              "
        f"{result['model']}"
    )

    print(
        f"Model Version:      "
        f"{result['model_version']}"
    )

    print()
    print("Class probabilities")
    print("-" * 60)

    print(
        f"DOWN:   "
        f"{result['probabilities']['DOWN'] * 100:.2f}%"
    )

    print(
        f"STABLE: "
        f"{result['probabilities']['STABLE'] * 100:.2f}%"
    )

    print(
        f"UP:     "
        f"{result['probabilities']['UP'] * 100:.2f}%"
    )

    print()
    print(
        result[
            "disclaimer"
        ]
    )

    print("=" * 60)


# ============================================================
# COMMAND LINE
# ============================================================

def main() -> None:

    if len(
        sys.argv
    ) != 3:

        print(
            "Usage:"
        )

        print(
            "python -m "
            "src.ml.inference.predictor "
            "<asset> <horizon>"
        )

        print()

        print(
            "Examples:"
        )

        print(
            "python -m "
            "src.ml.inference.predictor "
            "BTC 6"
        )

        print(
            "python -m "
            "src.ml.inference.predictor "
            "ethereum 12"
        )

        print(
            "python -m "
            "src.ml.inference.predictor "
            "SOL 24"
        )

        sys.exit(
            1
        )

    asset = (
        sys.argv[1]
    )

    horizon = (
        sys.argv[2]
    )

    try:

        result = predict_trend(
            asset,
            horizon,
        )

        print_prediction(
            result
        )

        print()
        print(
            "JSON response:"
        )

        print(
            json.dumps(
                result,
                indent=4,
            )
        )

    except Exception as error:

        print()
        print(
            "[ERROR]"
        )

        print(
            str(error)
        )

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()