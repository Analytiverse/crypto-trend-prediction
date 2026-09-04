from sqlalchemy import text

from src.database.connection import get_engine


EXPECTED_TABLES = {
    "coins",
    "market_history_raw",
    "market_hourly",
    "market_snapshots",
}


def verify_schema():
    """
    Verify that the expected database tables exist.
    """
    engine = get_engine()

    query = text(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
    )

    with engine.connect() as connection:
        result = connection.execute(query)

        tables = {row[0] for row in result}

    print("Tables found in database:")

    for table in sorted(tables):
        print(f"  - {table}")

    missing_tables = EXPECTED_TABLES - tables

    if missing_tables:
        print("\nMissing tables:")

        for table in sorted(missing_tables):
            print(f"  - {table}")

        raise RuntimeError("Database schema verification failed.")

    print("\nDatabase schema verification successful.")


if __name__ == "__main__":
    verify_schema()