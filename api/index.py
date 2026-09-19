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

import math
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.infrastructure.database.connection import engine
from src.repositories.prediction_repository import get_latest_predictions
from src.controllers.explanation_controller import (
    get_prediction_explanation,
)
from src.controllers.pipeline_controller import (
    run_daily_market_update,
)
from src.controllers.prediction_controller import get_prediction

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

def make_json_safe(value: Any) -> Any:
    """
    Recursively convert values that cannot be represented
    in strict JSON.

    PostgreSQL data is NOT modified.

    NaN / Infinity -> None
    Decimal NaN / Infinity -> None
    Normal Decimal values are preserved and FastAPI
    serializes them normally.
    """

    if isinstance(value, dict):
        return {
            key: make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, Decimal):
        if value.is_nan() or value.is_infinite():
            return None
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    return value


def rows_to_dicts(result) -> list[dict]:
    """
    Convert SQLAlchemy rows to dictionaries.
    """

    return [
        dict(row._mapping)
        for row in result
    ]


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
    Read existing AlphaPulse market data from PostgreSQL.

    This endpoint is READ ONLY.

    It does NOT:
    - fetch CoinGecko data
    - clean data
    - repair gaps
    - insert rows
    - update rows
    - delete rows
    - retrain models

    Dashboard mapping:

    Current Coin Market
        -> latest market_snapshots row for each coin

    Raw Market
        -> market_snapshots

    Raw History
        -> market_history_raw

    Clean Market
        -> latest market_hourly row for each coin

    Clean History
        -> market_hourly
    """

    # --------------------------------------------------------
    # CURRENT COIN MARKET
    # Latest real market snapshot for every tracked coin.
    # --------------------------------------------------------

    latest_snapshot_query = """
        SELECT DISTINCT ON (ms.coin_id)
            ms.coin_id,
            c.symbol,
            c.name,
            ms.timestamp,
            ms.current_price,
            ms.market_cap,
            ms.market_cap_rank,
            ms.total_volume,
            ms.high_24h,
            ms.low_24h,
            ms.price_change_24h,
            ms.price_change_percentage_24h,
            ms.circulating_supply,
            ms.total_supply,
            ms.max_supply,
            ms.fetched_at
        FROM market_snapshots AS ms
        INNER JOIN coins AS c
            ON c.coin_id = ms.coin_id
        ORDER BY
            ms.coin_id,
            ms.timestamp DESC
    """

    # --------------------------------------------------------
    # DATABASE COUNTS
    # These are FULL table counts, not the 500-row UI limit.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TRACKED COINS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RAW MARKET SNAPSHOTS
    # --------------------------------------------------------

    raw_market_query = """
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
        ORDER BY
            timestamp DESC,
            coin_id
        LIMIT 500
    """

    # --------------------------------------------------------
    # RAW HISTORICAL DATA
    # --------------------------------------------------------

    raw_history_query = """
        SELECT
            id,
            coin_id,
            timestamp,
            price,
            market_cap,
            total_volume,
            fetched_at
        FROM market_history_raw
        ORDER BY
            timestamp DESC,
            coin_id
        LIMIT 500
    """

    # --------------------------------------------------------
    # CLEAN MARKET
    # Latest cleaned hourly observation for every coin.
    # --------------------------------------------------------

    clean_market_query = """
        SELECT DISTINCT ON (mh.coin_id)
            mh.coin_id,
            c.symbol,
            c.name,
            mh.timestamp,
            mh.price,
            mh.market_cap,
            mh.total_volume,
            mh.created_at,
            mh.updated_at
        FROM market_hourly AS mh
        INNER JOIN coins AS c
            ON c.coin_id = mh.coin_id
        ORDER BY
            mh.coin_id,
            mh.timestamp DESC
    """

    # --------------------------------------------------------
    # CLEAN HISTORICAL DATA
    # --------------------------------------------------------

    clean_history_query = """
        SELECT
            coin_id,
            timestamp,
            price,
            market_cap,
            total_volume,
            created_at,
            updated_at
        FROM market_hourly
        ORDER BY
            timestamp DESC,
            coin_id
        LIMIT 500
    """

    # --------------------------------------------------------
    # DATA QUALITY / CLEANING SUMMARY
    #
    # These values are calculated directly from PostgreSQL.
    # Nothing is hardcoded.
    # --------------------------------------------------------

    quality_query = """
        SELECT

            (
                SELECT COUNT(*)
                FROM market_history_raw
                WHERE
                    coin_id IS NULL
                    OR timestamp IS NULL
                    OR price IS NULL
                    OR market_cap IS NULL
                    OR total_volume IS NULL
            ) AS raw_null_rows,

            (
                SELECT COALESCE(
                    SUM(duplicate_count - 1),
                    0
                )
                FROM (
                    SELECT
                        COUNT(*) AS duplicate_count
                    FROM market_history_raw
                    GROUP BY
                        coin_id,
                        timestamp
                    HAVING COUNT(*) > 1
                ) AS raw_duplicates
            ) AS raw_duplicate_rows,

            (
                SELECT COUNT(*)
                FROM market_hourly
                WHERE
                    coin_id IS NULL
                    OR timestamp IS NULL
                    OR price IS NULL
                    OR market_cap IS NULL
                    OR total_volume IS NULL
            ) AS clean_null_rows,

            (
                SELECT COALESCE(
                    SUM(duplicate_count - 1),
                    0
                )
                FROM (
                    SELECT
                        COUNT(*) AS duplicate_count
                    FROM market_hourly
                    GROUP BY
                        coin_id,
                        timestamp
                    HAVING COUNT(*) > 1
                ) AS clean_duplicates
            ) AS clean_duplicate_rows
    """

    try:
        with engine.connect() as connection:

            # ------------------------------------------------
            # Current/latest snapshot per coin
            # ------------------------------------------------

            latest_result = connection.exec_driver_sql(
                latest_snapshot_query
            )

            latest = rows_to_dicts(
                latest_result
            )

            # ------------------------------------------------
            # Full database counts
            # ------------------------------------------------

            counts_result = connection.exec_driver_sql(
                counts_query
            ).first()

            counts = (
                dict(counts_result._mapping)
                if counts_result
                else {}
            )

            # ------------------------------------------------
            # Coin metadata
            # ------------------------------------------------

            coins_result = connection.exec_driver_sql(
                coins_query
            )

            coins = rows_to_dicts(
                coins_result
            )

            # ------------------------------------------------
            # Raw market snapshots
            # ------------------------------------------------

            raw_market_result = connection.exec_driver_sql(
                raw_market_query
            )

            raw_market = rows_to_dicts(
                raw_market_result
            )

            # ------------------------------------------------
            # Raw historical data
            # ------------------------------------------------

            raw_history_result = connection.exec_driver_sql(
                raw_history_query
            )

            raw_history = rows_to_dicts(
                raw_history_result
            )

            # ------------------------------------------------
            # Latest clean market data
            # ------------------------------------------------

            clean_market_result = connection.exec_driver_sql(
                clean_market_query
            )

            clean_market = rows_to_dicts(
                clean_market_result
            )

            # ------------------------------------------------
            # Clean historical data
            # ------------------------------------------------

            clean_history_result = connection.exec_driver_sql(
                clean_history_query
            )

            clean_history = rows_to_dicts(
                clean_history_result
            )

            # ------------------------------------------------
            # Cleaning / quality statistics
            # ------------------------------------------------

            quality_result = connection.exec_driver_sql(
                quality_query
            ).first()

            quality = (
                dict(quality_result._mapping)
                if quality_result
                else {}
            )

        # ----------------------------------------------------
        # RESPONSE EXPECTED BY index.html
        # ----------------------------------------------------

        response = {
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

            # Current Coin Market
            "latest": latest,

            # Live cleaning/data-quality statistics
            "quality": {
                "raw_null_rows": quality.get(
                    "raw_null_rows",
                    0,
                ),
                "raw_duplicate_rows": quality.get(
                    "raw_duplicate_rows",
                    0,
                ),
                "clean_null_rows": quality.get(
                    "clean_null_rows",
                    0,
                ),
                "clean_duplicate_rows": quality.get(
                    "clean_duplicate_rows",
                    0,
                ),
            },

            # Raw database views
            "raw": {
                "coins": coins,
                "market_snapshots": raw_market,
                "market_history": raw_history,
            },

            # Clean database views
            "cleaned": {
                "market_snapshots": clean_market,
                "market_history": clean_history,
            },
        }

        # Critical fix:
        # PostgreSQL/driver data may contain NaN values.
        # Strict JSON cannot serialize NaN.
        # Convert only invalid JSON values to null.
        return make_json_safe(response)

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

        return make_json_safe({
            "status": "success",
            "count": len(rows),
            "predictions": rows,
        })

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
        result = get_prediction(

            asset=asset,
            horizon_hours=horizon,
       )

        return make_json_safe({
            "status": "success",
            "result": result,
        })

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
        result = get_prediction_explanation(
            asset=asset,
            horizon_hours=horizon,
       )

        return make_json_safe({
            "status": "success",
            "result": result,
        })

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
        result = run_daily_market_update()

        return make_json_safe({
            "status": "success",
            "result": result,
        })

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
