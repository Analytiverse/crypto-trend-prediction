from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.infrastructure.database.connection import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED_DIRECTORY = PROJECT_ROOT / "data" / "seed"

OUTPUT_FILE = SEED_DIRECTORY / "market_history_7d.csv"

EXPECTED_COINS = {
    "bitcoin",
    "ethereum",
    "solana",
    "ripple",
    "cardano",
}

SEED_HOURS = 168


def load_seed_data():
    """
    Export a common 7-day hourly window from market_hourly.

    A common timestamp window is used so all five cryptocurrencies
    are aligned to the same period.
    """
    engine = get_engine()

    latest_query = text(
        """
        SELECT MIN(latest_timestamp) AS common_latest_timestamp
        FROM (
            SELECT
                coin_id,
                MAX(timestamp) AS latest_timestamp
            FROM market_hourly
            WHERE coin_id IN (
                'bitcoin',
                'ethereum',
                'solana',
                'ripple',
                'cardano'
            )
            GROUP BY coin_id
        ) latest_per_coin
        """
    )

    with engine.connect() as connection:
        common_latest = connection.execute(
            latest_query
        ).scalar_one_or_none()

    if common_latest is None:
        raise RuntimeError(
            "No market_hourly data was found for seed export."
        )

    # 168 hourly observations means:
    # latest timestamp plus the previous 167 hours.
    start_timestamp = common_latest - pd.Timedelta(
        hours=SEED_HOURS - 1
    )

    data_query = text(
        """
        SELECT
            coin_id,
            timestamp,
            price,
            market_cap,
            total_volume
        FROM market_hourly
        WHERE coin_id IN (
            'bitcoin',
            'ethereum',
            'solana',
            'ripple',
            'cardano'
        )
          AND timestamp >= :start_timestamp
          AND timestamp <= :end_timestamp
        ORDER BY coin_id, timestamp
        """
    )

    df = pd.read_sql(
        data_query,
        engine,
        params={
            "start_timestamp": start_timestamp,
            "end_timestamp": common_latest,
        },
    )

    return df, start_timestamp, common_latest


def validate_seed_data(df):
    """
    Validate the exported seed before writing it to the repository.
    """
    if df.empty:
        raise RuntimeError(
            "Seed export returned no rows."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    actual_coins = set(
        df["coin_id"]
        .astype(str)
        .str.lower()
        .unique()
    )

    missing_coins = EXPECTED_COINS - actual_coins

    if missing_coins:
        raise RuntimeError(
            "Seed export is missing coin(s): "
            + ", ".join(sorted(missing_coins))
        )

    duplicate_count = df.duplicated(
        subset=["coin_id", "timestamp"]
    ).sum()

    if duplicate_count:
        raise RuntimeError(
            f"Seed export contains "
            f"{duplicate_count} duplicate coin/timestamp rows."
        )

    if df[
        [
            "coin_id",
            "timestamp",
            "price",
            "market_cap",
            "total_volume",
        ]
    ].isnull().any().any():
        raise RuntimeError(
            "Seed export contains null values."
        )

    counts = (
        df.groupby("coin_id")
        .size()
        .sort_index()
    )

    print("")
    print("Seed observations per coin:")

    for coin_id, count in counts.items():
        print(
            f"  {coin_id}: {count:,}"
        )

    insufficient = counts[
        counts < SEED_HOURS
    ]

    if not insufficient.empty:
        details = ", ".join(
            f"{coin_id}={count}"
            for coin_id, count in insufficient.items()
        )

        raise RuntimeError(
            "The selected 7-day window is incomplete. "
            f"Expected {SEED_HOURS} hourly observations "
            f"per coin. Found: {details}"
        )

    unexpected = counts[
        counts > SEED_HOURS
    ]

    if not unexpected.empty:
        raise RuntimeError(
            "Seed export contains more than the expected "
            "168 observations for at least one coin."
        )

    expected_total = (
        len(EXPECTED_COINS) * SEED_HOURS
    )

    if len(df) != expected_total:
        raise RuntimeError(
            f"Expected exactly {expected_total:,} seed rows, "
            f"but found {len(df):,}."
        )

    print("")
    print(
        f"[OK] Seed contains exactly "
        f"{expected_total:,} rows."
    )

    print(
        "[OK] Every supported coin contains exactly "
        "168 hourly observations."
    )

    print(
        "[OK] No duplicate coin/timestamp rows detected."
    )

    print(
        "[OK] No required null values detected."
    )


def export_seed_data():
    """
    Export and validate the static AlphaPulse 7-day bootstrap dataset.
    """
    print("")
    print("============================================================")
    print("ALPHAPULSE 7-DAY SEED EXPORT")
    print("============================================================")
    print("")

    SEED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[INFO] Reading clean hourly data from PostgreSQL...")

    df, start_timestamp, end_timestamp = load_seed_data()

    print(
        f"[INFO] Candidate rows loaded: "
        f"{len(df):,}"
    )

    print(
        f"[INFO] Seed window: "
        f"{start_timestamp} -> {end_timestamp}"
    )

    validate_seed_data(df)

    df = df[
        [
            "coin_id",
            "timestamp",
            "price",
            "market_cap",
            "total_volume",
        ]
    ].copy()

    df = df.sort_values(
        ["coin_id", "timestamp"]
    ).reset_index(drop=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("")
    print(
        f"[OK] Seed dataset saved to:\n"
        f"     {OUTPUT_FILE}"
    )

    print("")
    print("============================================================")
    print("[OK] 7-DAY SEED EXPORT COMPLETE")
    print("============================================================")
    print("")


if __name__ == "__main__":
    export_seed_data()