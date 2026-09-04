from sqlalchemy import text

from src.database.connection import get_engine


def verify_backfill():
    engine = get_engine()

    with engine.connect() as connection:

        coin_count = connection.execute(
            text("SELECT COUNT(*) FROM coins")
        ).scalar_one()

        raw_count = connection.execute(
            text("SELECT COUNT(*) FROM market_history_raw")
        ).scalar_one()

        hourly_count = connection.execute(
            text("SELECT COUNT(*) FROM market_hourly")
        ).scalar_one()

        raw_duplicates = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT
                        coin_id,
                        timestamp,
                        COUNT(*)
                    FROM market_history_raw
                    GROUP BY coin_id, timestamp
                    HAVING COUNT(*) > 1
                ) duplicates;
                """
            )
        ).scalar_one()

        hourly_duplicates = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT
                        coin_id,
                        timestamp,
                        COUNT(*)
                    FROM market_hourly
                    GROUP BY coin_id, timestamp
                    HAVING COUNT(*) > 1
                ) duplicates;
                """
            )
        ).scalar_one()

        rows_per_coin = connection.execute(
            text(
                """
                SELECT
                    coin_id,
                    COUNT(*) AS row_count,
                    MIN(timestamp) AS earliest_timestamp,
                    MAX(timestamp) AS latest_timestamp
                FROM market_hourly
                GROUP BY coin_id
                ORDER BY coin_id;
                """
            )
        ).fetchall()

    print("Database backfill verification")
    print("--------------------------------")
    print(f"Coins: {coin_count}")
    print(f"Raw rows: {raw_count}")
    print(f"Hourly rows: {hourly_count}")
    print(f"Raw duplicate timestamps: {raw_duplicates}")
    print(f"Hourly duplicate timestamps: {hourly_duplicates}")

    print("\nHourly data by coin:")

    for row in rows_per_coin:
        print(
            f"{row.coin_id}: "
            f"{row.row_count} rows | "
            f"{row.earliest_timestamp} -> "
            f"{row.latest_timestamp}"
        )


if __name__ == "__main__":
    verify_backfill()