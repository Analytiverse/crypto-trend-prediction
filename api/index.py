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

from src.database.connection import engine
from src.database.prediction_repository import get_latest_predictions
from src.llm.explanation_service import explain_prediction
from src.pipelines.daily_market_pipeline import run_daily_pipeline
from src.prediction.predictor import predict_trend


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
    Return the latest market data for all supported coins.

    Price, market cap, and volume come from market_hourly.

    Symbol and name come from the coins metadata table.
    """

    query = """
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

    try:
        with engine.connect() as connection:
            result = connection.exec_driver_sql(query)

            rows = [
                dict(row._mapping)
                for row in result
            ]

        return {
            "status": "success",
            "count": len(rows),
            "data": rows,
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

    Normally this returns the latest prediction for
    each coin/horizon combination.
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

    The Groq-hosted LLM only explains:
        - predicted trend
        - confidence
        - class probabilities
        - uncertainty

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