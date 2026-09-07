import os
from decimal import Decimal
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import text

from src.database.connection import get_engine
from src.pipelines.daily_market_pipeline import run_daily_pipeline


app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def serialize_value(value):
    if isinstance(value, Decimal):
        return float(value)

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return value


@app.get("/", response_class=HTMLResponse)
def dashboard():
    index_file = PROJECT_ROOT / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=500,
            detail="index.html not found.",
        )

    return index_file.read_text(encoding="utf-8")


@app.get("/api/market-data")
def get_market_data():
    engine = get_engine()

    try:
        with engine.connect() as connection:
            latest_rows = connection.execute(
                text(
                    """
                    SELECT DISTINCT ON (mh.coin_id)
                        mh.coin_id,
                        c.symbol,
                        c.name,
                        mh.price,
                        mh.market_cap,
                        mh.total_volume,
                        mh.timestamp
                    FROM market_hourly mh
                    JOIN coins c
                        ON c.coin_id = mh.coin_id
                    ORDER BY
                        mh.coin_id,
                        mh.timestamp DESC;
                    """
                )
            ).mappings().all()

            recent_rows = connection.execute(
                text(
                    """
                    SELECT
                        mh.coin_id,
                        c.symbol,
                        c.name,
                        mh.price,
                        mh.market_cap,
                        mh.total_volume,
                        mh.timestamp
                    FROM market_hourly mh
                    JOIN coins c
                        ON c.coin_id = mh.coin_id
                    ORDER BY mh.timestamp DESC
                    LIMIT 25;
                    """
                )
            ).mappings().all()

        return {
            "status": "success",
            "latest": [
                {
                    key: serialize_value(value)
                    for key, value in row.items()
                }
                for row in latest_rows
            ],
            "recent": [
                {
                    key: serialize_value(value)
                    for key, value in row.items()
                }
                for row in recent_rows
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get("/api/daily-ingestion")
def daily_ingestion(
    authorization: str | None = Header(default=None)
):
    cron_secret = os.getenv("CRON_SECRET")

    if not cron_secret:
        raise HTTPException(
            status_code=500,
            detail="CRON_SECRET is not configured.",
        )

    if authorization != f"Bearer {cron_secret}":
        raise HTTPException(
            status_code=401,
            detail="Unauthorized.",
        )

    try:
        return run_daily_pipeline()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )