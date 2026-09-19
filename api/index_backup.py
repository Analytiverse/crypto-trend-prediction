"""
AlphaPulse FastAPI Application

Endpoints:
    GET  /
    GET  /api/market-data
    GET  /api/predictions
    GET  /api/predict
    GET  /api/explain
    GET  /api/daily-ingestion
    GET  /api/health

The LLM explanation endpoint uses the existing AlphaPulse
ML prediction pipeline and the Groq explanation service.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.infrastructure.database.connection import engine
from src.repositories.prediction_repository import get_latest_predictions
from src.llm.explanation_service import explain_prediction
from src.pipelines.daily_market_pipeline import run_daily_pipeline
from src.ml.inference.predictor import predict_trend


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="AlphaPulse API",
    description=(
        "Crypto trend prediction and "
        "LLM explanation API."
    ),
    version="1.0.0",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_ROOT / "index.html"


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_ASSETS = {
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "ADA",
}

SUPPORTED_HORIZONS = {
    6,
    12,
    24,
}


# ============================================================
# HELPERS
# ============================================================

def validate_asset(asset: str) -> str:
    """
    Normalize and validate an asset symbol.
    """

    normalized_asset = asset.upper().strip()

    if normalized_asset not in SUPPORTED_ASSETS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported asset. "
                "Supported assets: "
                "BTC, ETH, SOL, XRP, ADA."
            ),
        )

    return normalized_asset


def validate_horizon(horizon: int) -> int:
    """
    Validate prediction horizon.
    """

    if horizon not in SUPPORTED_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported horizon. "
                "Supported horizons: "
                "6, 12, 24 hours."
            ),
        )

    return horizon


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def home():
    """
    Serve the AlphaPulse dashboard.
    """

    if not INDEX_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html was not found.",
        )

    return FileResponse(INDEX_FILE)


# ============================================================
# MARKET DATA
# ============================================================

@app.get("/api/market-data")
def market_data():
    """
    Return market information required by the dashboard.

    This endpoint only READS existing database data.
    It does not insert, update, delete, or modify data.
    """

    # Latest cleaned hourly record for each coin.
    latest_query = """
        SELECT DISTINCT ON (mh.coin_id)
            mh.coin_id,
            c.symbol,
            c.name,
            mh.timestamp,
            mh.price,
            mh.market_cap,
            mh.total_volume
        FROM market_hourly AS mh
        INNER JOIN coins AS c
            ON c.coin_id = mh.coin_id
        ORDER BY
            mh.coin_id,
            mh.timestamp DESC
    """

    # Actual database row counts.
    counts_query = """
        SELECT
            (SELECT COUNT(*) FROM coins)
                AS coins,

            (SELECT COUNT(*) FROM market_history_raw)
                AS raw_history,

            (SELECT COUNT(*) FROM market_hourly)
                AS clean_history,

            (SELECT COUNT(*) FROM market_snapshots)
                AS snapshots
    """

    # Coin metadata.
    coins_query = """
        SELECT
            coin_id,
            symbol,
            name,
            created_at,
            updated_at
        FROM coins
        ORDER BY symbol
    """

    # Raw historical CoinGecko records.
    raw_history_query = """
        SELECT
            coin_id,
            timestamp,
            price,
            market_cap,
            total_volume
        FROM market_history_raw
        ORDER BY timestamp DESC
        LIMIT 500
    """

    # Cleaned hourly records.
    clean_history_query = """
        SELECT
            coin_id,
            timestamp,
            price,
            market_cap,
            total_volume
        FROM market_hourly
        ORDER BY timestamp DESC
        LIMIT 500
    """

    # Raw CoinGecko market snapshots.
    #
    # IMPORTANT:
    # market_snapshots contains "current_price",
    # NOT a column named "price".
    snapshots_query = """
        SELECT
            id,
            coin_id,
            timestamp,
            current_price,
            market_cap,
            market_cap_rank,
            total_volume,
            high_24h,
            low_24h,
            price_change_24h,
            price_change_percentage_24h,
            circulating_supply,
            total_supply,
            max_supply,
            fetched_at
        FROM market_snapshots
        ORDER BY timestamp DESC
        LIMIT 500
    """

    try:
        with engine.connect() as connection:

            # ------------------------------------------------
            # Latest record for each coin
            # ------------------------------------------------

            latest_result = connection.exec_driver_sql(
                latest_query
            )

            latest = [
                dict(row._mapping)
                for row in latest_result
            ]

            # ------------------------------------------------
            # Counts
            # ------------------------------------------------

            counts_result = connection.exec_driver_sql(
                counts_query
            ).first()

            if counts_result:
                counts = dict(
                    counts_result._mapping
                )
            else:
                counts = {}

            # ------------------------------------------------
            # Coins
            # ------------------------------------------------

            coins_result = connection.exec_driver_sql(
                coins_query
            )

            coins = [
                dict(row._mapping)
                for row in coins_result
            ]

            # ------------------------------------------------
            # Raw history
            # ------------------------------------------------

            raw_history_result = (
                connection.exec_driver_sql(
                    raw_history_query
                )
            )

            raw_history = [
                dict(row._mapping)
                for row in raw_history_result
            ]

            # ------------------------------------------------
            # Clean hourly history
            # ------------------------------------------------

            clean_history_result = (
                connection.exec_driver_sql(
                    clean_history_query
                )
            )

            clean_history = [
                dict(row._mapping)
                for row in clean_history_result
            ]

            # ------------------------------------------------
            # Market snapshots
            # ------------------------------------------------

            snapshots_result = (
                connection.exec_driver_sql(
                    snapshots_query
                )
            )

            snapshots = [
                dict(row._mapping)
                for row in snapshots_result
            ]

        # ----------------------------------------------------
        # RESPONSE EXPECTED BY index.html
        # ----------------------------------------------------

        return {
            "status": "success",

            "counts": {
                "coins": counts.get(
                    "coins",
                    0,
                ),
                "raw_history": counts.get(
                    "raw_history",
                    0,
                ),
                "clean_history": counts.get(
                    "clean_history",
                    0,
                ),
                "snapshots": counts.get(
                    "snapshots",
                    0,
                ),
            },

            "latest": latest,

            "raw": {
                "coins": coins,
                "market_history": raw_history,
                "market_snapshots": snapshots,
            },

            "cleaned": {
                "market_history": clean_history,
                "market_snapshots": snapshots,
            },
        }

    except Exception as exc:
        print(
            f"Market data endpoint failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to load market data.",
        ) from exc


# ============================================================
# STORED PREDICTIONS
# ============================================================

@app.get("/api/predictions")
def predictions():
    """
    Return the latest stored predictions.
    """

    try:
        rows = get_latest_predictions()

        return {
            "status": "success",
            "count": len(rows),
            "predictions": rows,
        }

    except Exception as exc:
        print(
            f"Predictions endpoint failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to load predictions.",
        ) from exc


# ============================================================
# LIVE ML PREDICTION
# ============================================================

@app.get("/api/predict")
def predict(
    asset: str,
    horizon: int,
):
    """
    Generate one live AlphaPulse ML prediction.

    Example:
        /api/predict?asset=BTC&horizon=6
    """

    asset = validate_asset(asset)
    horizon = validate_horizon(horizon)

    try:
        result = predict_trend(
            asset=asset,
            horizon_hours=horizon,
        )

        return {
            "status": "success",
            "result": result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(
            f"Prediction failed for "
            f"{asset} {horizon}h: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate prediction.",
        ) from exc


# ============================================================
# GROQ LLM EXPLANATION
# ============================================================

@app.get("/api/explain")
def explain(
    asset: str,
    horizon: int,
):
    """
    Generate a grounded natural-language explanation
    for one AlphaPulse ML prediction.

    The ML model makes the prediction.
    Groq only explains the prediction.

    Example:
        /api/explain?asset=BTC&horizon=6
    """

    asset = validate_asset(asset)
    horizon = validate_horizon(horizon)

    try:
        result = explain_prediction(
            asset=asset,
            horizon_hours=horizon,
        )

        return {
            "status": "success",
            "result": result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(
            f"LLM explanation failed for "
            f"{asset} {horizon}h: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate "
                "prediction explanation."
            ),
        ) from exc


# ============================================================
# DAILY INGESTION
# ============================================================

@app.get("/api/daily-ingestion")
def daily_ingestion():
    """
    Run the AlphaPulse daily market pipeline.

    NOTE:
    CRON_SECRET authorization is not currently enforced
    by this endpoint.
    """

    try:
        result = run_daily_pipeline()

        return {
            "status": "success",
            "result": result,
        }

    except Exception as exc:
        print(
            f"Daily ingestion endpoint failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Daily ingestion pipeline failed.",
        ) from exc


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():
    """
    Simple API health check.
    """

    return {
        "status": "ok",
        "service": "AlphaPulse API",
        "llm_provider": "Groq",
    }