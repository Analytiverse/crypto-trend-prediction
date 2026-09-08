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


def serialize_rows(rows):
    return [
        {
            key: serialize_value(value)
            for key, value in row.items()
        }
        for row in rows
    ]


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

            # ---------------------------------------------------------
            # Dashboard latest coin values
            # ---------------------------------------------------------

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

            # ---------------------------------------------------------
            # 1. RAW COINS
            # ---------------------------------------------------------

            raw_coins = connection.execute(
                text(
                    """
                    SELECT
                        coin_id,
                        symbol,
                        name,
                        created_at,
                        updated_at
                    FROM coins
                    ORDER BY coin_id;
                    """
                )
            ).mappings().all()

            # ---------------------------------------------------------
            # 2. RAW MARKET HISTORY
            # ---------------------------------------------------------

            raw_history = connection.execute(
                text(
                    """
                    SELECT
                        id,
                        coin_id,
                        timestamp,
                        price,
                        market_cap,
                        total_volume,
                        fetched_at
                    FROM market_history_raw
                    ORDER BY timestamp DESC
                    LIMIT 100;
                    """
                )
            ).mappings().all()

            # ---------------------------------------------------------
            # 3. RAW MARKET SNAPSHOTS
            # ---------------------------------------------------------

            raw_snapshots = connection.execute(
                text(
                    """
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
                    LIMIT 100;
                    """
                )
            ).mappings().all()

            # ---------------------------------------------------------
            # 4. CLEANED COINS
            #
            # coins is already the canonical dimension table.
            # We expose it separately in the UI as the cleaned view.
            # ---------------------------------------------------------

            clean_coins = connection.execute(
                text(
                    """
                    SELECT
                        coin_id,
                        UPPER(symbol) AS symbol,
                        name,
                        updated_at
                    FROM coins
                    ORDER BY coin_id;
                    """
                )
            ).mappings().all()

            # ---------------------------------------------------------
            # 5. CLEANED MARKET HISTORY
            # ---------------------------------------------------------

            clean_history = connection.execute(
                text(
                    """
                    SELECT
                        coin_id,
                        timestamp,
                        price,
                        market_cap,
                        total_volume,
                        created_at,
                        updated_at
                    FROM market_hourly
                    ORDER BY timestamp DESC
                    LIMIT 100;
                    """
                )
            ).mappings().all()

            # ---------------------------------------------------------
            # 6. CLEANED SNAPSHOT VIEW
            #
            # Latest snapshot for each coin/hour.
            # This is a read-only derived view for the dashboard.
            # It does NOT create another database table.
            # ---------------------------------------------------------

            clean_snapshots = connection.execute(
                text(
                    """
                    SELECT DISTINCT ON (
                        coin_id,
                        DATE_TRUNC('hour', timestamp)
                    )
                        coin_id,
                        DATE_TRUNC(
                            'hour',
                            timestamp
                        ) AS timestamp,
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
                        max_supply
                    FROM market_snapshots
                    ORDER BY
                        coin_id,
                        DATE_TRUNC('hour', timestamp) DESC,
                        timestamp DESC
                    LIMIT 100;
                    """
                )
            ).mappings().all()

            # ---------------------------------------------------------
            # Counts
            # ---------------------------------------------------------

            counts = connection.execute(
                text(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM coins)
                            AS coins,
                        (SELECT COUNT(*) FROM market_history_raw)
                            AS raw_history,
                        (SELECT COUNT(*) FROM market_hourly)
                            AS clean_history,
                        (SELECT COUNT(*) FROM market_snapshots)
                            AS snapshots;
                    """
                )
            ).mappings().one()

        return {
            "status": "success",

            "latest": serialize_rows(latest_rows),

            "counts": {
                key: serialize_value(value)
                for key, value in counts.items()
            },

            "raw": {
                "coins": serialize_rows(raw_coins),
                "market_history": serialize_rows(raw_history),
                "market_snapshots": serialize_rows(raw_snapshots),
            },

            "cleaned": {
                "coins": serialize_rows(clean_coins),
                "market_history": serialize_rows(clean_history),
                "market_snapshots": serialize_rows(clean_snapshots),
            },
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