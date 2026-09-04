from sqlalchemy import text

from src.database.connection import get_engine


def upsert_raw_history(df):
    """
    Insert or update raw CoinGecko historical observations.
    """
    if df.empty:
        return 0

    records = df[
        [
            "coin_id",
            "timestamp",
            "price",
            "market_cap",
            "total_volume",
        ]
    ].to_dict(orient="records")

    query = text(
        """
        INSERT INTO market_history_raw (
            coin_id,
            timestamp,
            price,
            market_cap,
            total_volume
        )
        VALUES (
            :coin_id,
            :timestamp,
            :price,
            :market_cap,
            :total_volume
        )
        ON CONFLICT (coin_id, timestamp)
        DO UPDATE SET
            price = EXCLUDED.price,
            market_cap = EXCLUDED.market_cap,
            total_volume = EXCLUDED.total_volume,
            fetched_at = NOW();
        """
    )

    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(query, records)

    return len(records)


def upsert_hourly_history(df):
    """
    Insert or update cleaned hourly observations.
    """
    if df.empty:
        return 0

    records = df[
        [
            "coin_id",
            "timestamp",
            "price",
            "market_cap",
            "total_volume",
        ]
    ].to_dict(orient="records")

    query = text(
        """
        INSERT INTO market_hourly (
            coin_id,
            timestamp,
            price,
            market_cap,
            total_volume
        )
        VALUES (
            :coin_id,
            :timestamp,
            :price,
            :market_cap,
            :total_volume
        )
        ON CONFLICT (coin_id, timestamp)
        DO UPDATE SET
            price = EXCLUDED.price,
            market_cap = EXCLUDED.market_cap,
            total_volume = EXCLUDED.total_volume,
            updated_at = NOW();
        """
    )

    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(query, records)

    return len(records)