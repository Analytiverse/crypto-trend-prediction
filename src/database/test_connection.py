from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.database.connection import get_engine


def test_database_connection() -> None:
    engine = get_engine()

    try:
        with engine.connect() as connection:
            version = connection.execute(
                text("SELECT version();")
            ).scalar()

            database = connection.execute(
                text("SELECT current_database();")
            ).scalar()

        print("Database connection successful")
        print(f"Database: {database}")
        print(f"PostgreSQL version: {version}")

    except SQLAlchemyError as exc:
        print("Database connection failed")
        print(f"Error: {exc}")
        raise


if __name__ == "__main__":
    test_database_connection()