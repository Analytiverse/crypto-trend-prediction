from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import DATABASE_URL


# Explicitly use the Psycopg 3 driver with SQLAlchemy
SQLALCHEMY_DATABASE_URL = DATABASE_URL.replace(
    "postgresql://",
    "postgresql+psycopg://",
    1,
)

engine: Engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
)


def get_engine() -> Engine:
    """
    Return the shared SQLAlchemy database engine.
    """
    return engine