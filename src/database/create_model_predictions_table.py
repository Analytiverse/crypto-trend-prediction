from sqlalchemy import text

from src.infrastructure.database.connection import get_engine


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS model_predictions (
    id BIGSERIAL PRIMARY KEY,

    coin_id VARCHAR(50) NOT NULL,

    prediction_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    data_timestamp TIMESTAMPTZ NOT NULL,

    horizon_hours INTEGER NOT NULL,

    predicted_class VARCHAR(16) NOT NULL,

    prob_down DOUBLE PRECISION NOT NULL,
    prob_stable DOUBLE PRECISION NOT NULL,
    prob_up DOUBLE PRECISION NOT NULL,

    confidence DOUBLE PRECISION NOT NULL,

    current_price NUMERIC(30, 10) NOT NULL,

    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,

    actual_future_price NUMERIC(30, 10),
    actual_return DOUBLE PRECISION,
    actual_class VARCHAR(16),

    is_evaluated BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_model_predictions_horizon
        CHECK (horizon_hours IN (6, 12, 24)),

    CONSTRAINT chk_model_predictions_class
        CHECK (predicted_class IN ('DOWN', 'STABLE', 'UP')),

    CONSTRAINT chk_model_predictions_prob_down
        CHECK (prob_down >= 0 AND prob_down <= 1),

    CONSTRAINT chk_model_predictions_prob_stable
        CHECK (prob_stable >= 0 AND prob_stable <= 1),

    CONSTRAINT chk_model_predictions_prob_up
        CHECK (prob_up >= 0 AND prob_up <= 1),

    CONSTRAINT chk_model_predictions_confidence
        CHECK (confidence >= 0 AND confidence <= 1),

    CONSTRAINT uq_model_prediction
        UNIQUE (
            coin_id,
            data_timestamp,
            horizon_hours,
            model_version
        )
);
"""


CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_model_predictions_lookup
ON model_predictions (
    coin_id,
    horizon_hours,
    data_timestamp DESC
);
"""


def create_model_predictions_table():
    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(text(CREATE_TABLE_SQL))
        connection.execute(text(CREATE_INDEX_SQL))

    print("model_predictions table is ready.")


if __name__ == "__main__":
    create_model_predictions_table()
    