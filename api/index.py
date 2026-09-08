import math
import os

from decimal import Decimal
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import text


app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================================================
# JSON SERIALIZATION HELPERS
# =========================================================

def serialize_value(value):
    """
    Convert database values into JSON-safe values.

    Handles:
    - Decimal
    - datetime/date values
    - NaN
    - positive/negative infinity
    """

    if value is None:
        return None

    if isinstance(value, Decimal):
        value = float(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return value


def serialize_rows(rows):
    """
    Convert SQLAlchemy mapping rows
    into JSON-safe dictionaries.
    """

    return [
        {
            key: serialize_value(value)
            for key, value in row.items()
        }
        for row in rows
    ]


# =========================================================
# FRONTEND DASHBOARD
# =========================================================

@app.get("/", response_class=HTMLResponse)
def dashboard():

    index_file = PROJECT_ROOT / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=500,
            detail="index.html not found.",
        )

    return index_file.read_text(
        encoding="utf-8"
    )


# =========================================================
# MARKET DATA API
# =========================================================

@app.get("/api/market-data")
def get_market_data():

    try:
        from src.database.connection import get_engine

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Database initialization failed: "
                f"{str(exc)}"
            ),
        )

    try:
        engine = get_engine()

        with engine.connect() as connection:

            # =================================================
            # LATEST CLEAN HOURLY DATA FOR EACH COIN
            # =================================================

            latest_rows = connection.execute(
                text(
                    """
                    SELECT DISTINCT ON (mh.coin_id)
                        mh.coin_id,
                        UPPER(c.symbol) AS symbol,
                        c.name,
                        mh.price,
                        mh.market_cap,
                        mh.total_volume,
                        mh.timestamp
                    FROM market_hourly mh
                    JOIN coins c
                        ON c.coin_id = mh.coin_id
                    WHERE mh.price IS NOT NULL
                      AND mh.market_cap IS NOT NULL
                      AND mh.total_volume IS NOT NULL
                    ORDER BY
                        mh.coin_id,
                        mh.timestamp DESC;
                    """
                )
            ).mappings().all()

            # =================================================
            # RAW COINS
            # =================================================

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

            # =================================================
            # RAW MARKET HISTORY
            # =================================================

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

            # =================================================
            # RAW CURRENT MARKET SNAPSHOTS
            # =================================================

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

            # =================================================
            # CLEAN COINS
            # =================================================

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

            # =================================================
            # CLEAN HOURLY HISTORY
            # =================================================

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
                    WHERE price IS NOT NULL
                      AND market_cap IS NOT NULL
                      AND total_volume IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT 100;
                    """
                )
            ).mappings().all()

            # =================================================
            # CLEAN CURRENT MARKET SNAPSHOTS
            #
            # One latest snapshot per coin per hour.
            # Rows missing price, market cap or volume
            # are removed.
            # =================================================

            clean_snapshots = connection.execute(
                text(
                    """
                    SELECT
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
                        max_supply
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

                    ) AS snapshot_rows

                    WHERE row_num = 1
                      AND current_price IS NOT NULL
                      AND market_cap IS NOT NULL
                      AND total_volume IS NOT NULL

                    ORDER BY timestamp DESC

                    LIMIT 100;
                    """
                )
            ).mappings().all()

            # =================================================
            # FINAL MODEL-READY / MERGED TABLE
            # =================================================

            merged_market = connection.execute(
                text(
                    """
                    SELECT
                        mh.coin_id,
                        UPPER(c.symbol) AS symbol,
                        c.name,
                        mh.timestamp,
                        mh.price,
                        mh.market_cap,
                        mh.total_volume
                    FROM market_hourly mh
                    JOIN coins c
                        ON c.coin_id = mh.coin_id
                    WHERE mh.price IS NOT NULL
                      AND mh.market_cap IS NOT NULL
                      AND mh.total_volume IS NOT NULL
                    ORDER BY
                        mh.timestamp DESC,
                        mh.coin_id
                    LIMIT 100;
                    """
                )
            ).mappings().all()

            # =================================================
            # DATA QUALITY COUNTS
            # =================================================

            quality = connection.execute(
                text(
                    """
                    SELECT

                        (
                            SELECT COUNT(*)
                            FROM market_history_raw
                            WHERE price IS NULL
                               OR market_cap IS NULL
                               OR total_volume IS NULL
                        ) AS raw_history_null_rows,

                        (
                            SELECT
                                COUNT(*)
                                -
                                COUNT(
                                    DISTINCT (
                                        coin_id,
                                        timestamp
                                    )
                                )
                            FROM market_history_raw
                        ) AS raw_history_duplicate_rows,

                        (
                            SELECT COUNT(*)
                            FROM market_hourly
                            WHERE price IS NULL
                               OR market_cap IS NULL
                               OR total_volume IS NULL
                        ) AS clean_history_null_rows,

                        (
                            SELECT
                                COUNT(*)
                                -
                                COUNT(
                                    DISTINCT (
                                        coin_id,
                                        timestamp
                                    )
                                )
                            FROM market_hourly
                        ) AS clean_history_duplicate_rows,

                        (
                            SELECT COUNT(*)
                            FROM market_snapshots
                            WHERE current_price IS NULL
                               OR market_cap IS NULL
                               OR total_volume IS NULL
                        ) AS raw_snapshot_null_rows;

                    """
                )
            ).mappings().one()

            # =================================================
            # TABLE COUNTS
            # =================================================

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

        # =====================================================
        # RESPONSE
        # =====================================================

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

            "merged": serialize_rows(
                merged_market
            ),

            "quality": {
                key: serialize_value(value)
                for key, value in quality.items()
            },
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Market data query failed: "
                f"{str(exc)}"
            ),
        )


# =========================================================
# DAILY INGESTION ENDPOINT
# =========================================================

@app.get("/api/daily-ingestion")
def daily_ingestion(
    authorization: str | None = Header(
        default=None
    )
):

    cron_secret = os.getenv(
        "CRON_SECRET"
    )

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
        from src.pipelines.daily_market_pipeline import (
            run_daily_pipeline,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Ingestion initialization failed: "
                f"{str(exc)}"
            ),
        )

    try:
        return run_daily_pipeline()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Daily ingestion failed: "
                f"{str(exc)}"
            ),
        )