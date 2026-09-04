from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.database.connection import get_engine


RAW_FILE = Path("data/raw/market_history.csv")
HOURLY_FILE = Path("data/processed/market_hourly.csv")


COINS = [
    {
        "coin_id": "bitcoin",
        "symbol": "btc",
        "name": "Bitcoin",
    },
    {
        "coin_id": "ethereum",
        "symbol": "eth",
        "name": "Ethereum",
    },
    {
        "coin_id": "solana",
        "symbol": "sol",
        "name": "Solana",
    },
    {
        "coin_id": "cardano",
        "symbol": "ada",
        "name": "Cardano",
    },
    {
        "coin_id": "ripple",
        "symbol": "xrp",
        "name": "XRP",
    },
]


def load_csv_files():
    """
    Load existing historical CSV files.
    """
    raw_df = pd.read_csv(
        RAW_FILE,
        parse_dates=["timestamp"],
    )

    hourly_df = pd.read_csv(
        HOURLY_FILE,
        parse_dates=["timestamp"],
    )

    return raw_df, hourly_df


def insert_coins(connection):
    """
    Insert tracked cryptocurrencies into the coins table.
    Existing coins are updated safely.
    """

    query = text(
        """
        INSERT INTO coins (
            coin_id,
            symbol,
            name
        )
        VALUES (
            :coin_id,
            :symbol,
            :name
        )
        ON CONFLICT (coin_id)
        DO UPDATE SET
            symbol = EXCLUDED.symbol,
            name = EXCLUDED.name,
            updated_at = NOW();
        """
    )

    connection.execute(query, COINS)


def backfill_raw(connection, raw_df):
    """
    Backfill historical raw market data.
    """

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
            total_volume = EXCLUDED.total_volume;
        """
    )

    records = raw_df[
        [
            "coin_id",
            "timestamp",
            "price",
            "market_cap",
            "total_volume",
        ]
    ].to_dict(orient="records")

    connection.execute(query, records)


def backfill_hourly(connection, hourly_df):
    """
    Backfill cleaned hourly market data.
    """

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

    records = hourly_df[
        [
            "coin_id",
            "timestamp",
            "price",
            "market_cap",
            "total_volume",
        ]
    ].to_dict(orient="records")

    connection.execute(query, records)


def run_backfill():
    print("Loading CSV files...")

    raw_df, hourly_df = load_csv_files()

    print(f"Raw rows found: {len(raw_df):,}")
    print(f"Hourly rows found: {len(hourly_df):,}")

    engine = get_engine()

    print("Starting database backfill...")

    with engine.begin() as connection:
        print("Inserting coins...")
        insert_coins(connection)

        print("Backfilling market_history_raw...")
        backfill_raw(connection, raw_df)

        print("Backfilling market_hourly...")
        backfill_hourly(connection, hourly_df)

    print("Historical backfill completed successfully.")


if __name__ == "__main__":
    run_backfill()