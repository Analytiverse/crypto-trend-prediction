import os
from decimal import Decimal
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import text


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


# ---------------------------------------------------------
# DASHBOARD UI
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def dashboard():
    index_file = PROJECT_ROOT / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=500,
            detail="index.html not found.",
        )

    return index_file.read_text(encoding="utf-8")


# ---------------------------------------------------------
# MARKET DATA API
# ---------------------------------------------------------

@app.get("/api/market-data")
def get_market_data():

    # Import only when this endpoint is called.
    # This prevents DB/config issues from crashing the whole app.
    try:
        from src.database.connection import get_engine
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database initialization failed: {str(exc)}",
        )

    try:
        engine = get_engine()

        with engine.connect() as connection:

            # -------------------------------------------------
            # Latest market value for each coin
            # -------------------------------------------------

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


            # -------------------------------------------------
            # RAW COINS
            # -------------------------------------------------

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


            # -------------------------------------------------
            # RAW MARKET HISTORY
            # -------------------------------------------------

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


            # -------------------------------------------------
            # RAW MARKET SNAPSHOTS
            # -------------------------------------------------

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


            # -------------------------------------------------
            # CLEAN COINS VIEW
            # -------------------------------------------------

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


            # -------------------------------------------------
            # CLEAN MARKET HISTORY
            # -------------------------------------------------

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


            # -------------------------------------------------
            # CLEAN SNAPSHOT VIEW
            #
            # One latest snapshot per coin/hour.
            # -------------------------------------------------

            clean_snapshots = connection.execute(
                text(
                    """
                    SELECT *
                    FROM (
                        SELECT
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
                            max_supply,

                            ROW_NUMBER() OVER (
                                PARTITION BY
                                    coin_id,
                                    DATE_TRUNC(
                                        'hour',
                                        timestamp
                                    )
                                ORDER BY timestamp DESC
                            ) AS row_num

                        FROM market_snapshots
                    ) snapshot_rows

                    WHERE row_num = 1

                    ORDER BY timestamp DESC

                    LIMIT 100;
                    """
                )
            ).mappings().all()


            # -------------------------------------------------
            # TABLE COUNTS
            # -------------------------------------------------

            counts = connection.execute(
                text(
                    """
                    SELECT
                        (
                            SELECT COUNT(*)
                            FROM coins
                        ) AS coins,

                        (
                            SELECT COUNT(*)
                            FROM market_history_raw
                        ) AS raw_history,

                        (
                            SELECT COUNT(*)
                            FROM market_hourly
                        ) AS clean_history,

                        (
                            SELECT COUNT(*)
                            FROM market_snapshots
                        ) AS snapshots;
                    """
                )
            ).mappings().one()


        return {
            "status": "success",

            "latest": serialize_rows(
                latest_rows
            ),

            "counts": {
                key: serialize_value(value)
                for key, value in counts.items()
            },

            "raw": {
                "coins": serialize_rows(
                    raw_coins
                ),
                "market_history": serialize_rows(
                    raw_history
                ),
                "market_snapshots": serialize_rows(
                    raw_snapshots
                ),
            },

            "cleaned": {
                "coins": serialize_rows(
                    clean_coins
                ),
                "market_history": serialize_rows(
                    clean_history
                ),
                "market_snapshots": serialize_rows(
                    clean_snapshots
                ),
            },
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Market data query failed: {str(exc)}",
        )


# ---------------------------------------------------------
# DAILY INGESTION
# ---------------------------------------------------------

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

    # Import ingestion pipeline only when cron calls this route.
    try:
        from src.pipelines.daily_market_pipeline import (
            run_daily_pipeline
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion initialization failed: {str(exc)}",
        )

    try:
        return run_daily_pipeline()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Daily ingestion failed: {str(exc)}",
        )