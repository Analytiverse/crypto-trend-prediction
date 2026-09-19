"""
AlphaPulse Prediction Repository

Responsibilities:
- Create the predictions table if it does not exist.
- Store batch model predictions in PostgreSQL.
- Upsert predictions safely so rerunning the same timestamp/horizon
  does not create duplicates.
- Read the latest predictions for API/UI use.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from src.infrastructure.database.connection import get_engine


# ---------------------------------------------------------------------
# Create predictions table
# ---------------------------------------------------------------------

def ensure_predictions_table() -> None:
    """
    Create the production predictions table if it does not already exist.

    One prediction is uniquely identified by:

        coin_id
        prediction_timestamp
        horizon_hours
        model_version
    """

    engine = get_engine()

    create_table_query = text(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id BIGSERIAL PRIMARY KEY,

            coin_id VARCHAR(50) NOT NULL,
            asset VARCHAR(20) NOT NULL,

            prediction_timestamp TIMESTAMPTZ NOT NULL,
            horizon_hours INTEGER NOT NULL,

            current_price DOUBLE PRECISION NOT NULL,

            predicted_trend VARCHAR(20) NOT NULL,
            model_confidence DOUBLE PRECISION NOT NULL,

            probability_down DOUBLE PRECISION NOT NULL,
            probability_stable DOUBLE PRECISION NOT NULL,
            probability_up DOUBLE PRECISION NOT NULL,

            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(50) NOT NULL,

            threshold DOUBLE PRECISION NOT NULL,
            feature_count INTEGER NOT NULL,

            generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            CONSTRAINT uq_predictions
            UNIQUE (
                coin_id,
                prediction_timestamp,
                horizon_hours,
                model_version
            )
        );
        """
    )

    create_timestamp_index = text(
        """
        CREATE INDEX IF NOT EXISTS
        idx_predictions_timestamp
        ON predictions (
            prediction_timestamp DESC
        );
        """
    )

    create_asset_horizon_index = text(
        """
        CREATE INDEX IF NOT EXISTS
        idx_predictions_asset_horizon
        ON predictions (
            coin_id,
            horizon_hours,
            prediction_timestamp DESC
        );
        """
    )

    with engine.begin() as connection:
        connection.execute(create_table_query)
        connection.execute(create_timestamp_index)
        connection.execute(create_asset_horizon_index)


# ---------------------------------------------------------------------
# Store one prediction
# ---------------------------------------------------------------------

def upsert_prediction(
    prediction: dict[str, Any],
) -> None:
    """
    Insert one prediction.

    If the same coin + timestamp + horizon + model version
    already exists, update it instead of creating a duplicate.
    """

    probabilities = prediction.get(
        "probabilities",
        {},
    )

    query = text(
        """
        INSERT INTO predictions (
            coin_id,
            asset,
            prediction_timestamp,
            horizon_hours,
            current_price,
            predicted_trend,
            model_confidence,
            probability_down,
            probability_stable,
            probability_up,
            model_name,
            model_version,
            threshold,
            feature_count
        )
        VALUES (
            :coin_id,
            :asset,
            :prediction_timestamp,
            :horizon_hours,
            :current_price,
            :predicted_trend,
            :model_confidence,
            :probability_down,
            :probability_stable,
            :probability_up,
            :model_name,
            :model_version,
            :threshold,
            :feature_count
        )

        ON CONFLICT (
            coin_id,
            prediction_timestamp,
            horizon_hours,
            model_version
        )

        DO UPDATE SET
            asset = EXCLUDED.asset,
            current_price = EXCLUDED.current_price,
            predicted_trend = EXCLUDED.predicted_trend,
            model_confidence = EXCLUDED.model_confidence,
            probability_down = EXCLUDED.probability_down,
            probability_stable = EXCLUDED.probability_stable,
            probability_up = EXCLUDED.probability_up,
            model_name = EXCLUDED.model_name,
            threshold = EXCLUDED.threshold,
            feature_count = EXCLUDED.feature_count,
            updated_at = NOW();
        """
    )

    values = {
        "coin_id": prediction["coin_id"],
        "asset": prediction["asset"],
        "prediction_timestamp": prediction[
            "prediction_timestamp"
        ],
        "horizon_hours": prediction[
            "prediction_horizon_hours"
        ],
        "current_price": prediction[
            "current_price"
        ],
        "predicted_trend": prediction[
            "predicted_trend"
        ],
        "model_confidence": prediction[
            "model_confidence"
        ],
        "probability_down": probabilities[
            "DOWN"
        ],
        "probability_stable": probabilities[
            "STABLE"
        ],
        "probability_up": probabilities[
            "UP"
        ],
        "model_name": prediction[
            "model"
        ],
        "model_version": prediction[
            "model_version"
        ],
        "threshold": prediction[
            "threshold"
        ],
        "feature_count": prediction[
            "feature_count"
        ],
    }

    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(
            query,
            values,
        )


# ---------------------------------------------------------------------
# Store complete batch
# ---------------------------------------------------------------------

def upsert_prediction_batch(
    batch_response: dict[str, Any],
) -> int:
    """
    Store all successful predictions from batch_predictor.

    Failed predictions are ignored.

    Returns:
        Number of predictions written.
    """

    ensure_predictions_table()

    predictions = batch_response.get(
        "predictions",
        [],
    )

    successful_predictions = [
        prediction
        for prediction in predictions
        if prediction.get("status") != "FAILED"
    ]

    if not successful_predictions:
        return 0

    engine = get_engine()

    query = text(
        """
        INSERT INTO predictions (
            coin_id,
            asset,
            prediction_timestamp,
            horizon_hours,
            current_price,
            predicted_trend,
            model_confidence,
            probability_down,
            probability_stable,
            probability_up,
            model_name,
            model_version,
            threshold,
            feature_count
        )
        VALUES (
            :coin_id,
            :asset,
            :prediction_timestamp,
            :horizon_hours,
            :current_price,
            :predicted_trend,
            :model_confidence,
            :probability_down,
            :probability_stable,
            :probability_up,
            :model_name,
            :model_version,
            :threshold,
            :feature_count
        )

        ON CONFLICT (
            coin_id,
            prediction_timestamp,
            horizon_hours,
            model_version
        )

        DO UPDATE SET
            asset = EXCLUDED.asset,
            current_price = EXCLUDED.current_price,
            predicted_trend = EXCLUDED.predicted_trend,
            model_confidence = EXCLUDED.model_confidence,
            probability_down = EXCLUDED.probability_down,
            probability_stable = EXCLUDED.probability_stable,
            probability_up = EXCLUDED.probability_up,
            model_name = EXCLUDED.model_name,
            threshold = EXCLUDED.threshold,
            feature_count = EXCLUDED.feature_count,
            updated_at = NOW();
        """
    )

    records = []

    for prediction in successful_predictions:

        probabilities = prediction[
            "probabilities"
        ]

        records.append(
            {
                "coin_id": prediction[
                    "coin_id"
                ],
                "asset": prediction[
                    "asset"
                ],
                "prediction_timestamp": prediction[
                    "prediction_timestamp"
                ],
                "horizon_hours": prediction[
                    "prediction_horizon_hours"
                ],
                "current_price": prediction[
                    "current_price"
                ],
                "predicted_trend": prediction[
                    "predicted_trend"
                ],
                "model_confidence": prediction[
                    "model_confidence"
                ],
                "probability_down": probabilities[
                    "DOWN"
                ],
                "probability_stable": probabilities[
                    "STABLE"
                ],
                "probability_up": probabilities[
                    "UP"
                ],
                "model_name": prediction[
                    "model"
                ],
                "model_version": prediction[
                    "model_version"
                ],
                "threshold": prediction[
                    "threshold"
                ],
                "feature_count": prediction[
                    "feature_count"
                ],
            }
        )

    with engine.begin() as connection:
        connection.execute(
            query,
            records,
        )

    return len(records)


# ---------------------------------------------------------------------
# Read latest predictions
# ---------------------------------------------------------------------

def get_latest_predictions() -> list[dict[str, Any]]:
    """
    Return the newest prediction for every:

        coin × horizon

    This will later be used by the API/UI.
    """

    ensure_predictions_table()

    query = text(
        """
        SELECT DISTINCT ON (
            coin_id,
            horizon_hours
        )
            id,
            coin_id,
            asset,
            prediction_timestamp,
            horizon_hours,
            current_price,
            predicted_trend,
            model_confidence,
            probability_down,
            probability_stable,
            probability_up,
            model_name,
            model_version,
            threshold,
            feature_count,
            generated_at,
            updated_at
        FROM predictions

        ORDER BY
            coin_id,
            horizon_hours,
            prediction_timestamp DESC,
            updated_at DESC;
        """
    )

    engine = get_engine()

    with engine.connect() as connection:
        rows = (
            connection.execute(query)
            .mappings()
            .all()
        )

    return [
        dict(row)
        for row in rows
    ]


# ---------------------------------------------------------------------
# Count stored predictions
# ---------------------------------------------------------------------

def count_predictions() -> int:
    """
    Return total number of prediction rows stored.
    """

    ensure_predictions_table()

    engine = get_engine()

    query = text(
        """
        SELECT COUNT(*)
        FROM predictions;
        """
    )

    with engine.connect() as connection:
        count = connection.execute(
            query
        ).scalar_one()

    return int(count)