from pathlib import Path

from sqlalchemy import text

from src.database.connection import get_engine


def initialize_database():
    """
    Execute schema.sql and create the database tables.
    """
    schema_path = Path(__file__).with_name("schema.sql")

    schema_sql = schema_path.read_text(encoding="utf-8")

    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(text(schema_sql))

    print("Database schema initialized successfully.")


if __name__ == "__main__":
    initialize_database()