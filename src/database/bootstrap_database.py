from pathlib import Path

import pandas as pd
from sqlalchemy import inspect, text

from src.database.connection import get_engine
from src.database.init_db import initialize_database
from src.database.backfill import (
    insert_coins,
    backfill_raw,
    backfill_hourly,
)
from src.processing.clean_history import clean_history_dataframe


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED_DIRECTORY = PROJECT_ROOT / "data" / "seed"

SEED_FILE = SEED_DIRECTORY / "market_history_7d.csv"

REQUIRED_TABLES = {
    "coins",
    "market_history_raw",
    "market_hourly",
    "market_snapshots",
}

EXPECTED_COINS = {
    "bitcoin",
    "ethereum",
    "solana",
    "ripple",
    "cardano",
}

MINIMUM_HOURS_PER_COIN = 120


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def get_existing_tables():
    """
    Return the names of tables currently available in PostgreSQL.
    """
    engine = get_engine()
    inspector = inspect(engine)

    return set(inspector.get_table_names())


def ensure_schema():
    """
    Create the AlphaPulse database schema when required.

    schema.sql uses CREATE TABLE IF NOT EXISTS, so this operation
    is safe to run repeatedly.
    """
    existing_tables = get_existing_tables()
    missing_tables = REQUIRED_TABLES - existing_tables

    if not missing_tables:
        print("[OK] Required database tables already exist.")
        return

    print(
        "[INFO] Missing database tables: "
        + ", ".join(sorted(missing_tables))
    )

    print("[INFO] Initializing database schema...")

    initialize_database()

    existing_tables = get_existing_tables()
    missing_tables = REQUIRED_TABLES - existing_tables

    if missing_tables:
        raise RuntimeError(
            "Database initialization completed, but the following "
            "required tables are still missing: "
            + ", ".join(sorted(missing_tables))
        )

    print("[OK] Database schema is ready.")


def get_database_counts():
    """
    Return current row counts for the main AlphaPulse tables.
    """
    engine = get_engine()

    queries = {
        "coins": "SELECT COUNT(*) FROM coins",
        "market_history_raw":
            "SELECT COUNT(*) FROM market_history_raw",
        "market_hourly":
            "SELECT COUNT(*) FROM market_hourly",
        "market_snapshots":
            "SELECT COUNT(*) FROM market_snapshots",
    }

    counts = {}

    with engine.connect() as connection:
        for table_name, query in queries.items():
            counts[table_name] = connection.execute(
                text(query)
            ).scalar_one()

    return counts


def print_database_counts(counts):
    """
    Print database counts in a readable format.
    """
    print("")
    print("Current database state:")
    print(f"  Coins:              {counts['coins']:,}")
    print(
        f"  Raw history:        "
        f"{counts['market_history_raw']:,}"
    )
    print(
        f"  Clean hourly:       "
        f"{counts['market_hourly']:,}"
    )
    print(
        f"  Market snapshots:   "
        f"{counts['market_snapshots']:,}"
    )


def database_has_market_data():
    """
    Determine whether the database already contains usable
    AlphaPulse hourly market history.

    If market_hourly contains data, bootstrap seeding is skipped.
    """
    engine = get_engine()

    query = text(
        """
        SELECT COUNT(*)
        FROM market_hourly
        """
    )

    with engine.connect() as connection:
        row_count = connection.execute(query).scalar_one()

    return row_count > 0


def validate_seed_file():
    """
    Validate that the bundled seed CSV exists and has the columns
    required by the existing AlphaPulse ingestion/cleaning logic.
    """
    if not SEED_FILE.exists():
        raise FileNotFoundError(
            "\n7-day seed file was not found.\n"
            f"Expected location:\n{SEED_FILE}\n"
        )

    seed_df = pd.read_csv(SEED_FILE)

    required_columns = {
        "coin_id",
        "timestamp",
        "price",
        "market_cap",
        "total_volume",
    }

    missing_columns = required_columns - set(seed_df.columns)

    if missing_columns:
        raise ValueError(
            "Seed file is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    if seed_df.empty:
        raise ValueError(
            "Seed file exists but contains no market data."
        )

    return seed_df


def validate_seed_coins(seed_df):
    """
    Ensure the seed contains all five supported cryptocurrencies.
    """
    actual_coins = set(
        seed_df["coin_id"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
    )

    missing_coins = EXPECTED_COINS - actual_coins

    if missing_coins:
        raise ValueError(
            "Seed dataset is missing required coin(s): "
            + ", ".join(sorted(missing_coins))
        )

    print("[OK] Seed dataset contains all 5 supported coins.")


def prepare_seed_data(seed_df):
    """
    Prepare raw and cleaned hourly seed datasets using the same
    cleaning function used by the main AlphaPulse pipeline.
    """
    raw_df = seed_df.copy()

    raw_df["coin_id"] = (
        raw_df["coin_id"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    raw_df["timestamp"] = pd.to_datetime(
        raw_df["timestamp"],
        utc=True,
    )

    raw_df = raw_df[
        [
            "coin_id",
            "timestamp",
            "price",
            "market_cap",
            "total_volume",
        ]
    ].copy()

    raw_df = raw_df.sort_values(
        ["coin_id", "timestamp"]
    )

    raw_df = raw_df.drop_duplicates(
        subset=["coin_id", "timestamp"],
        keep="last",
    )

    raw_df = raw_df.reset_index(drop=True)

    hourly_df = clean_history_dataframe(raw_df)

    if hourly_df.empty:
        raise ValueError(
            "Seed data produced no valid hourly observations "
            "after cleaning."
        )

    return raw_df, hourly_df


def validate_history_length(hourly_df):
    """
    Verify that each cryptocurrency has enough historical rows
    for production feature generation.

    The predictor loads approximately 120 hours of recent history,
    while the longest rolling feature uses 72 hours.
    """
    counts = (
        hourly_df
        .groupby("coin_id")
        .size()
        .to_dict()
    )

    print("")
    print("Seed hourly observations:")

    insufficient = []

    for coin_id in sorted(EXPECTED_COINS):
        count = counts.get(coin_id, 0)

        print(f"  {coin_id}: {count:,}")

        if count < MINIMUM_HOURS_PER_COIN:
            insufficient.append(
                f"{coin_id} ({count} rows)"
            )

    if insufficient:
        raise ValueError(
            "Seed dataset does not contain enough hourly history "
            "for: "
            + ", ".join(insufficient)
            + f". Minimum required: "
            f"{MINIMUM_HOURS_PER_COIN} rows per coin."
        )

    print(
        f"[OK] Every coin has at least "
        f"{MINIMUM_HOURS_PER_COIN} hourly observations."
    )


def seed_database(raw_df, hourly_df):
    """
    Insert the static seed dataset using the project's existing
    idempotent backfill functions.

    Existing matching coin/timestamp rows are updated rather than
    duplicated because the existing SQL uses ON CONFLICT.
    """
    engine = get_engine()

    print("")
    print("[INFO] Starting fresh-database seed...")

    with engine.begin() as connection:

        print("[INFO] Inserting supported coins...")
        insert_coins(connection)

        print(
            f"[INFO] Seeding {len(raw_df):,} raw "
            f"historical observations..."
        )
        backfill_raw(connection, raw_df)

        print(
            f"[INFO] Seeding {len(hourly_df):,} cleaned "
            f"hourly observations..."
        )
        backfill_hourly(connection, hourly_df)

    print("[OK] Seed data inserted successfully.")


def verify_database_after_seed():
    """
    Verify the database after seed insertion.
    """
    engine = get_engine()

    query = text(
        """
        SELECT
            coin_id,
            COUNT(*) AS row_count,
            MIN(timestamp) AS first_timestamp,
            MAX(timestamp) AS last_timestamp
        FROM market_hourly
        GROUP BY coin_id
        ORDER BY coin_id
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    found_coins = {
        row["coin_id"]
        for row in rows
    }

    missing_coins = EXPECTED_COINS - found_coins

    if missing_coins:
        raise RuntimeError(
            "Database bootstrap verification failed. "
            "Missing coin(s): "
            + ", ".join(sorted(missing_coins))
        )

    print("")
    print("Database hourly history after bootstrap:")

    for row in rows:
        print(
            f"  {row['coin_id']}: "
            f"{row['row_count']:,} rows | "
            f"{row['first_timestamp']} -> "
            f"{row['last_timestamp']}"
        )

    insufficient = [
        row["coin_id"]
        for row in rows
        if row["row_count"] < MINIMUM_HOURS_PER_COIN
    ]

    if insufficient:
        raise RuntimeError(
            "Database bootstrap completed but insufficient "
            "history exists for: "
            + ", ".join(sorted(insufficient))
        )

    print("")
    print("[OK] Database bootstrap verification passed.")


# ------------------------------------------------------------
# Main bootstrap
# ------------------------------------------------------------

def bootstrap_database():
    """
    Prepare PostgreSQL for AlphaPulse.

    Behavior:

    1. Ensure required tables exist.
    2. Inspect current database state.
    3. If market_hourly already contains data, leave it untouched.
    4. If the database is empty, load the bundled 7-day seed.
    5. Use existing project cleaning/backfill logic.
    6. Verify that enough hourly history exists for predictions.

    This function is intentionally idempotent.
    """
    print("")
    print("============================================================")
    print("ALPHAPULSE DATABASE BOOTSTRAP")
    print("============================================================")
    print("")

    ensure_schema()

    counts = get_database_counts()
    print_database_counts(counts)

    if database_has_market_data():
        print("")
        print(
            "[OK] Existing market data detected. "
            "Seed bootstrap is not required."
        )
        print(
            "[OK] Existing PostgreSQL data was left unchanged."
        )
        return {
            "status": "existing_data",
            "seeded": False,
            "counts": counts,
        }

    print("")
    print("[INFO] No hourly market history was found.")
    print("[INFO] Fresh database bootstrap is required.")

    print("")
    print(
        f"[INFO] Loading bundled seed dataset:\n"
        f"       {SEED_FILE}"
    )

    seed_df = validate_seed_file()

    print(
        f"[OK] Seed file loaded: "
        f"{len(seed_df):,} raw rows."
    )

    validate_seed_coins(seed_df)

    raw_df, hourly_df = prepare_seed_data(seed_df)

    print(
        f"[OK] Clean hourly seed rows: "
        f"{len(hourly_df):,}"
    )

    validate_history_length(hourly_df)

    seed_database(
        raw_df=raw_df,
        hourly_df=hourly_df,
    )

    verify_database_after_seed()

    final_counts = get_database_counts()
    print_database_counts(final_counts)

    print("")
    print("============================================================")
    print("[OK] FRESH DATABASE BOOTSTRAP COMPLETE")
    print("============================================================")
    print("")

    return {
        "status": "seeded",
        "seeded": True,
        "counts": final_counts,
    }


if __name__ == "__main__":
    bootstrap_database()