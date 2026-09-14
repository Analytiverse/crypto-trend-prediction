from datetime import datetime, timezone

from sqlalchemy import text

from src.database.connection import get_engine


UPSERT_PREDICTION_SQL = """
INSERT INTO model_predictions (
    coin_id,
    prediction_timestamp,
    data_timestamp,
    horizon_hours,
    predicted_class,
    prob_down,
    prob_stable,
    prob_up,
    confidence,
    current_price,
    model_name,
    model_version,
    is_evaluated
)
VALUES (
    :coin_id,
    :prediction_timestamp,
    :data_timestamp,
    :horizon_hours,
    :predicted_class,
    :prob_down,
    :prob_stable,
    :prob_up,
    :confidence,
    :current_price,
    :model_name,
    :model_version,
    FALSE
)
ON CONFLICT (
    coin_id,
    data_timestamp,
    horizon_hours,
    model_version
)
DO UPDATE SET
    prediction_timestamp = EXCLUDED.prediction_timestamp,
    predicted_class = EXCLUDED.predicted_class,
    prob_down = EXCLUDED.prob_down,
    prob_stable = EXCLUDED.prob_stable,
    prob_up = EXCLUDED.prob_up,
    confidence = EXCLUDED.confidence,
    current_price = EXCLUDED.current_price,
    model_name = EXCLUDED.model_name
RETURNING id;
"""


def save_prediction(prediction: dict) -> int:
    """
    Save one model prediction.

    The predictor currently returns its latest usable market timestamp
    under the key 'prediction_timestamp'. We store that value as
    data_timestamp.

    prediction_timestamp in the database means the actual time the
    prediction was generated.
    """

    probabilities = prediction["probabilities"]

    params = {
        "coin_id": prediction["coin_id"],
        "prediction_timestamp": datetime.now(timezone.utc),
        "data_timestamp": prediction["prediction_timestamp"],
        "horizon_hours": prediction["prediction_horizon_hours"],
        "predicted_class": prediction["predicted_trend"],
        "prob_down": probabilities["DOWN"],
        "prob_stable": probabilities["STABLE"],
        "prob_up": probabilities["UP"],
        "confidence": prediction["model_confidence"],
        "current_price": prediction["current_price"],
        "model_name": prediction["model"],
        "model_version": prediction["model_version"],
    }

    engine = get_engine()

    with engine.begin() as connection:
        prediction_id = connection.execute(
            text(UPSERT_PREDICTION_SQL),
            params,
        ).scalar_one()

    return prediction_id


def get_recent_predictions(limit: int = 100):
    query = text(
        """
        SELECT
            id,
            coin_id,
            prediction_timestamp,
            data_timestamp,
            horizon_hours,
            predicted_class,
            prob_down,
            prob_stable,
            prob_up,
            confidence,
            current_price,
            model_name,
            model_version,
            actual_future_price,
            actual_return,
            actual_class,
            is_evaluated,
            created_at
        FROM model_predictions
        ORDER BY prediction_timestamp DESC
        LIMIT :limit;
        """
    )

    engine = get_engine()

    with engine.connect() as connection:
        rows = connection.execute(
            query,
            {"limit": limit},
        ).mappings().all()

    return [dict(row) for row in rows]