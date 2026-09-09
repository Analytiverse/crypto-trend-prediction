import pandas as pd
from sqlalchemy import text

from src.config import COINS
from src.database.connection import get_engine
from src.database.repository import (
    upsert_hourly_history,
    upsert_raw_history,
)
from src.ingestion.fetch_history import (
    fetch_coin_history_range,
    transform_history,
)
from src.processing.clean_history import (
    clean_history_dataframe,
)


def load_hourly_timestamps(
    lookback_days=None,
):
    """
    Load hourly timestamps currently stored in market_hourly.

    If lookback_days is provided, only recent data is checked.
    Otherwise the whole table is checked.
    """

    engine = get_engine()

    if lookback_days is None:
        query = text(
            """
            SELECT
                coin_id,
                timestamp
            FROM market_hourly
            ORDER BY coin_id, timestamp
            """
        )

        params = {}

    else:
        query = text(
            """
            SELECT
                coin_id,
                timestamp
            FROM market_hourly
            WHERE timestamp >= NOW() - (:lookback_days * INTERVAL '1 day')
            ORDER BY coin_id, timestamp
            """
        )

        params = {
            "lookback_days": lookback_days
        }

    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params=params,
        )

    if not df.empty:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
        )

    return df


def detect_gaps(
    lookback_days=None,
):
    """
    Detect missing hourly periods for each coin.

    Returns a list of dictionaries describing each gap.
    """

    df = load_hourly_timestamps(
        lookback_days=lookback_days
    )

    gaps = []

    if df.empty:
        return gaps

    for coin_id in COINS:

        coin_df = (
            df[
                df["coin_id"] == coin_id
            ]
            .sort_values("timestamp")
            .copy()
        )

        if coin_df.empty:
            continue

        coin_df["previous_timestamp"] = (
            coin_df["timestamp"].shift(1)
        )

        coin_df["time_diff"] = (
            coin_df["timestamp"]
            - coin_df["previous_timestamp"]
        )

        gap_rows = coin_df[
            coin_df["time_diff"]
            > pd.Timedelta(hours=1)
        ]

        for _, row in gap_rows.iterrows():

            previous_timestamp = (
                row["previous_timestamp"]
            )

            current_timestamp = (
                row["timestamp"]
            )

            missing_start = (
                previous_timestamp
                + pd.Timedelta(hours=1)
            )

            missing_end = (
                current_timestamp
                - pd.Timedelta(hours=1)
            )

            missing_hours = int(
                (
                    current_timestamp
                    - previous_timestamp
                )
                / pd.Timedelta(hours=1)
            ) - 1

            gaps.append(
                {
                    "coin_id": coin_id,
                    "previous_timestamp":
                        previous_timestamp,
                    "next_timestamp":
                        current_timestamp,
                    "missing_start":
                        missing_start,
                    "missing_end":
                        missing_end,
                    "missing_hours":
                        missing_hours,
                }
            )

    return gaps


def print_gaps(
    gaps,
    title,
):
    """
    Display detected gaps in a readable format.
    """

    print(
        "\n" + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )

    if not gaps:
        print(
            "No hourly gaps found."
        )

        return

    for gap in gaps:

        print(
            f"{gap['coin_id']}: "
            f"{gap['missing_hours']} missing hours | "
            f"{gap['missing_start']} "
            f"-> "
            f"{gap['missing_end']}"
        )

    print(
        f"\nTotal gap blocks: {len(gaps)}"
    )


def repair_gap(
    gap,
):
    """
    Attempt to repair one missing hourly range.
    """

    coin_id = gap["coin_id"]

    start_timestamp = (
        gap["missing_start"]
    )

    end_timestamp = (
        gap["missing_end"]
    )

    print(
        f"\nRepairing {coin_id}"
    )

    print(
        f"Missing range: "
        f"{start_timestamp} -> {end_timestamp}"
    )

    try:

        # Add a small boundary around the missing range.
        #
        # This helps ensure CoinGecko returns observations
        # close to the first and last missing hour.
        fetch_start = (
            start_timestamp
            - pd.Timedelta(hours=1)
        )

        fetch_end = (
            end_timestamp
            + pd.Timedelta(hours=1)
        )

        raw_response = (
            fetch_coin_history_range(
                coin_id=coin_id,
                start_timestamp=fetch_start,
                end_timestamp=fetch_end,
            )
        )

        if not raw_response:
            print(
                f"{coin_id}: "
                f"no data returned by CoinGecko."
            )

            return False

        raw_df = transform_history(
            coin_id=coin_id,
            data=raw_response,
        )

        if raw_df.empty:
            print(
                f"{coin_id}: "
                f"API response produced no rows."
            )

            return False

        # Keep raw API observations in the raw table.
        raw_rows = upsert_raw_history(
            raw_df
        )

        clean_df = clean_history_dataframe(
            raw_df
        )

        if clean_df.empty:
            print(
                f"{coin_id}: "
                f"no valid hourly rows after cleaning."
            )

            return False

        # Only keep rows relevant to the missing range.
        clean_df = clean_df[
            (
                clean_df["timestamp"]
                >= start_timestamp
            )
            &
            (
                clean_df["timestamp"]
                <= end_timestamp
            )
        ].copy()

        if clean_df.empty:
            print(
                f"{coin_id}: "
                f"CoinGecko did not return "
                f"the missing hourly observations."
            )

            return False

        hourly_rows = upsert_hourly_history(
            clean_df
        )

        print(
            f"{coin_id}: "
            f"{raw_rows} raw rows upserted, "
            f"{hourly_rows} repair rows upserted."
        )

        return True

    except Exception as exc:

        print(
            f"{coin_id}: "
            f"repair failed - {exc}"
        )

        return False


def repair_detected_gaps(
    lookback_days=None,
):
    """
    Detect and repair all gaps in the selected period.

    Returns a summary dictionary.
    """

    gaps_before = detect_gaps(
        lookback_days=lookback_days
    )

    print_gaps(
        gaps_before,
        "GAPS BEFORE REPAIR",
    )

    successful_repairs = 0

    failed_repairs = 0

    for gap in gaps_before:

        success = repair_gap(
            gap
        )

        if success:
            successful_repairs += 1

        else:
            failed_repairs += 1

    gaps_after = detect_gaps(
        lookback_days=lookback_days
    )

    print_gaps(
        gaps_after,
        "GAPS AFTER REPAIR",
    )

    print(
        "\n========== GAP REPAIR SUMMARY =========="
    )

    print(
        f"Gaps before: "
        f"{len(gaps_before)}"
    )

    print(
        f"Successful repair attempts: "
        f"{successful_repairs}"
    )

    print(
        f"Failed repair attempts: "
        f"{failed_repairs}"
    )

    print(
        f"Gaps remaining: "
        f"{len(gaps_after)}"
    )

    return {
        "gaps_before":
            len(gaps_before),

        "successful_repairs":
            successful_repairs,

        "failed_repairs":
            failed_repairs,

        "gaps_after":
            len(gaps_after),
    }


def main():
    """
    Run a full historical gap repair.

    This scans all stored hourly history, not only recent data.
    """

    print(
        "Starting historical hourly gap repair..."
    )

    repair_detected_gaps(
        lookback_days=None
    )


if __name__ == "__main__":
    main()