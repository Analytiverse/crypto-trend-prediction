from sqlalchemy import text

from src.database.connection import get_engine


def upsert_raw_history(df):
    """
    Insert or update raw market history rows.
    """

    if df is None or df.empty:
        return 0

    engine = get_engine()

    query = text(
        """
        INSERT INTO market_history_raw (
            coin_id,
            timestamp,
            price,
            market_cap,
            total_volume,
            fetched_at
        )
        VALUES (
            :coin_id,
            :timestamp,
            :price,
            :market_cap,
            :total_volume,
            NOW()
        )
        ON CONFLICT (coin_id, timestamp)
        DO UPDATE SET
            price = EXCLUDED.price,
            market_cap = EXCLUDED.market_cap,
            total_volume = EXCLUDED.total_volume,
            fetched_at = NOW()
        """
    )

    records = []

    for _, row in df.iterrows():
        records.append(
            {
                "coin_id": row["coin_id"],
                "timestamp": row["timestamp"],
                "price": row["price"],
                "market_cap": row["market_cap"],
                "total_volume": row["total_volume"],
            }
        )

    with engine.begin() as connection:
        connection.execute(query, records)

    return len(records)


def upsert_hourly_history(df):
    """
    Insert or update cleaned hourly market history rows.
    """

    if df is None or df.empty:
        return 0

    engine = get_engine()

    query = text(
        """
        INSERT INTO market_hourly (
            coin_id,
            timestamp,
            price,
            market_cap,
            total_volume,
            created_at,
            updated_at
        )
        VALUES (
            :coin_id,
            :timestamp,
            :price,
            :market_cap,
            :total_volume,
            NOW(),
            NOW()
        )
        ON CONFLICT (coin_id, timestamp)
        DO UPDATE SET
            price = EXCLUDED.price,
            market_cap = EXCLUDED.market_cap,
            total_volume = EXCLUDED.total_volume,
            updated_at = NOW()
        """
    )

    records = []

    for _, row in df.iterrows():
        records.append(
            {
                "coin_id": row["coin_id"],
                "timestamp": row["timestamp"],
                "price": row["price"],
                "market_cap": row["market_cap"],
                "total_volume": row["total_volume"],
            }
        )

    with engine.begin() as connection:
        connection.execute(query, records)

    return len(records)


def upsert_market_snapshots(df):
    """
    Insert or update current market snapshot rows.
    """

    if df is None or df.empty:
        return 0

    engine = get_engine()

    query = text(
        """
        INSERT INTO market_snapshots (
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
        )
        VALUES (
            :coin_id,
            :timestamp,
            :current_price,
            :market_cap,
            :market_cap_rank,
            :total_volume,
            :high_24h,
            :low_24h,
            :price_change_24h,
            :price_change_percentage_24h,
            :circulating_supply,
            :total_supply,
            :max_supply,
            NOW()
        )
        ON CONFLICT (coin_id, timestamp)
        DO UPDATE SET
            current_price = EXCLUDED.current_price,
            market_cap = EXCLUDED.market_cap,
            market_cap_rank = EXCLUDED.market_cap_rank,
            total_volume = EXCLUDED.total_volume,
            high_24h = EXCLUDED.high_24h,
            low_24h = EXCLUDED.low_24h,
            price_change_24h = EXCLUDED.price_change_24h,
            price_change_percentage_24h =
                EXCLUDED.price_change_percentage_24h,
            circulating_supply = EXCLUDED.circulating_supply,
            total_supply = EXCLUDED.total_supply,
            max_supply = EXCLUDED.max_supply,
            fetched_at = NOW()
        """
    )

    records = []

    for _, row in df.iterrows():
        records.append(
            {
                "coin_id": row["coin_id"],
                "timestamp": row["last_updated"],
                "current_price": row["current_price"],
                "market_cap": row["market_cap"],
                "market_cap_rank": row["market_cap_rank"],
                "total_volume": row["total_volume"],
                "high_24h": row["high_24h"],
                "low_24h": row["low_24h"],
                "price_change_24h": row["price_change_24h"],
                "price_change_percentage_24h":
                    row["price_change_percentage_24h"],
                "circulating_supply":
                    row["circulating_supply"],
                "total_supply": row["total_supply"],
                "max_supply": row["max_supply"],
            }
        )

    with engine.begin() as connection:
        connection.execute(query, records)

    return len(records)