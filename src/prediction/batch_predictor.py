"""
AlphaPulse Batch Predictor

Runs production predictions for all supported assets and horizons
and optionally persists successful predictions to PostgreSQL.

Usage:
    python -m src.prediction.batch_predictor
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any

from src.repositories.prediction_repository import (
    upsert_prediction_batch,
)

from src.prediction.predictor import (
    predict_trend,
)


# ============================================================
# CONFIGURATION
# ============================================================

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "ADA",
]

HORIZONS = [
    6,
    12,
    24,
]


# ============================================================
# SINGLE PREDICTION
# ============================================================

def run_single_prediction(
    asset: str,
    horizon: int,
) -> dict[str, Any]:
    """
    Generate one prediction directly through the Python
    prediction service.

    No subprocess is created.
    """

    return predict_trend(
        asset=asset,
        horizon_hours=horizon,
    )


# ============================================================
# BATCH PREDICTIONS
# ============================================================

def run_batch_predictions(
    persist: bool = False,
) -> dict[str, Any]:
    """
    Generate production predictions for all configured
    assets and horizons.

    Parameters
    ----------
    persist:
        When True, successful predictions are also
        upserted into PostgreSQL.

    Returns
    -------
    dict
        Complete batch prediction response.
    """

    predictions: list[
        dict[str, Any]
    ] = []

    total_predictions = (
        len(ASSETS)
        * len(HORIZONS)
    )

    print()
    print("=" * 70)
    print(
        "ALPHAPULSE - BATCH PRODUCTION PREDICTIONS"
    )
    print("=" * 70)
    print()

    print(
        f"Assets:      {', '.join(ASSETS)}"
    )

    print(
        "Horizons:    "
        + ", ".join(
            f"{horizon}h"
            for horizon in HORIZONS
        )
    )

    print(
        f"Predictions: {total_predictions}"
    )

    print()

    counter = 0

    for asset in ASSETS:

        for horizon in HORIZONS:

            counter += 1

            print(
                f"[{counter:02d}/{total_predictions}] "
                f"Predicting {asset} - "
                f"{horizon}h..."
            )

            try:

                prediction = (
                    run_single_prediction(
                        asset=asset,
                        horizon=horizon,
                    )
                )

                predictions.append(
                    prediction
                )

                trend = (
                    prediction.get(
                        "predicted_trend",
                        "UNKNOWN",
                    )
                )

                confidence = (
                    prediction.get(
                        "model_confidence_percent",
                        0.0,
                    )
                )

                print(
                    f"       -> {trend} "
                    f"({confidence:.2f}%)"
                )

            except Exception as exc:

                print(
                    f"       -> FAILED: {exc}"
                )

                predictions.append(
                    {
                        "asset":
                            asset,

                        "prediction_horizon_hours":
                            horizon,

                        "status":
                            "FAILED",

                        "error":
                            str(exc),
                    }
                )

    # ========================================================
    # TIMESTAMP VALIDATION
    # ========================================================

    valid_timestamps = [
        prediction.get(
            "prediction_timestamp"
        )
        for prediction in predictions
        if prediction.get(
            "prediction_timestamp"
        )
    ]

    unique_timestamps = sorted(
        set(
            valid_timestamps
        )
    )

    common_timestamp = None

    if len(
        unique_timestamps
    ) == 1:

        common_timestamp = (
            unique_timestamps[0]
        )

    # ========================================================
    # SUCCESS / FAILURE COUNTS
    # ========================================================

    successful = [
        prediction
        for prediction in predictions
        if prediction.get(
            "status"
        ) != "FAILED"
    ]

    failed = [
        prediction
        for prediction in predictions
        if prediction.get(
            "status"
        ) == "FAILED"
    ]

    batch_response = {

        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "prediction_timestamp":
            common_timestamp,

        "assets":
            ASSETS,

        "horizons":
            HORIZONS,

        "total_predictions":
            len(
                predictions
            ),

        "successful_predictions":
            len(
                successful
            ),

        "failed_predictions":
            len(
                failed
            ),

        "timestamp_consistent":
            len(
                unique_timestamps
            ) == 1,

        "predictions":
            predictions,
    }

    # ========================================================
    # OPTIONAL DATABASE PERSISTENCE
    # ========================================================

    if persist:

        print()
        print("=" * 70)
        print(
            "STORING PREDICTIONS"
        )
        print("=" * 70)

        stored_count = (
            upsert_prediction_batch(
                batch_response
            )
        )

        batch_response[
            "stored_predictions"
        ] = stored_count

        print(
            f"Stored predictions: "
            f"{stored_count}"
        )

    return batch_response


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    batch_response: dict[str, Any],
) -> None:
    """
    Print compact prediction summary.
    """

    predictions = (
        batch_response[
            "predictions"
        ]
    )

    print()
    print("=" * 70)
    print(
        "BATCH PREDICTION SUMMARY"
    )
    print("=" * 70)

    print(
        f"{'ASSET':<8}"
        f"{'HORIZON':<10}"
        f"{'TREND':<10}"
        f"{'CONFIDENCE':<12}"
        f"{'DOWN':<10}"
        f"{'STABLE':<10}"
        f"{'UP':<10}"
    )

    print("-" * 70)

    for prediction in predictions:

        if (
            prediction.get(
                "status"
            )
            == "FAILED"
        ):

            print(
                f"{prediction['asset']:<8}"
                f"{str(prediction['prediction_horizon_hours']) + 'h':<10}"
                f"{'FAILED':<10}"
            )

            continue

        probabilities = (
            prediction.get(
                "probabilities",
                {},
            )
        )

        asset = (
            prediction.get(
                "asset",
                "N/A",
            )
        )

        horizon = (
            str(
                prediction.get(
                    "prediction_horizon_hours",
                    "",
                )
            )
            + "h"
        )

        trend = (
            prediction.get(
                "predicted_trend",
                "N/A",
            )
        )

        confidence = (
            prediction.get(
                "model_confidence_percent",
                0,
            )
        )

        down = (
            probabilities.get(
                "DOWN",
                0,
            )
            * 100
        )

        stable = (
            probabilities.get(
                "STABLE",
                0,
            )
            * 100
        )

        up = (
            probabilities.get(
                "UP",
                0,
            )
            * 100
        )

        print(
            f"{asset:<8}"
            f"{horizon:<10}"
            f"{trend:<10}"
            f"{confidence:<12.2f}"
            f"{down:<10.2f}"
            f"{stable:<10.2f}"
            f"{up:<10.2f}"
        )

    print()

    print(
        "Successful predictions:",
        batch_response[
            "successful_predictions"
        ],
    )

    print(
        "Failed predictions:",
        batch_response[
            "failed_predictions"
        ],
    )

    print(
        "Prediction timestamp:",
        batch_response[
            "prediction_timestamp"
        ],
    )

    print(
        "Timestamp consistent:",
        batch_response[
            "timestamp_consistent"
        ],
    )

    if (
        "stored_predictions"
        in batch_response
    ):

        print(
            "Stored predictions:",
            batch_response[
                "stored_predictions"
            ],
        )


# ============================================================
# COMMAND LINE
# ============================================================

def main() -> None:

    try:

        batch_response = (
            run_batch_predictions(
                persist=True,
            )
        )

        print_summary(
            batch_response
        )

        print()
        print("=" * 70)
        print(
            "JSON RESPONSE"
        )
        print("=" * 70)

        print(
            json.dumps(
                batch_response,
                indent=4,
            )
        )

    except KeyboardInterrupt:

        print()
        print(
            "Batch prediction cancelled."
        )

        sys.exit(
            1
        )

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "BATCH PREDICTION FAILED"
        )
        print("=" * 70)

        print(
            exc
        )

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()