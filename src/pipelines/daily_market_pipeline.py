from src.config import COINS

from src.database.repository import (
    upsert_hourly_history,
    upsert_raw_history,
    upsert_market_snapshots,
)

from src.ingestion.fetch_history import (
    fetch_coin_history,
    transform_history,
)

from src.ingestion.fetch_market_snapshot import (
    fetch_market_data,
    transform_market_data,
)

from src.processing.clean_history import (
    clean_history_dataframe,
)


# We fetch the most recent 2 days of hourly history
# so the daily pipeline can safely catch any missing observations.
RECENT_HISTORY_DAYS = 2


def process_market_snapshots():
    """
    Fetch the latest /coins/markets data for all configured coins
    and store it in the market_snapshots table.
    """

    print("\nFetching current market snapshots...")

    try:
        raw_data = fetch_market_data()

        if not raw_data:
            print("No market snapshot data returned.")
            return 0

        snapshot_df = transform_market_data(raw_data)

        if snapshot_df is None or snapshot_df.empty:
            print("Snapshot DataFrame is empty.")
            return 0

        print(
            f"Fetched {len(snapshot_df)} current market snapshot rows."
        )

        rows_upserted = upsert_market_snapshots(
            snapshot_df
        )

        print(
            f"{rows_upserted} market snapshot rows upserted."
        )

        return rows_upserted

    except Exception as exc:
        print(
            f"Market snapshot processing failed: {exc}"
        )
        return 0


def process_coin(coin_id):
    """
    Fetch recent historical data for one coin,
    store raw data, clean it, and store hourly data.
    """

    print(f"\nProcessing {coin_id}...")

    try:
        # -----------------------------------------
        # 1. Fetch recent CoinGecko history
        # -----------------------------------------
        raw_response = fetch_coin_history(
            coin_id,
            days=RECENT_HISTORY_DAYS,
        )

        # -----------------------------------------
        # 2. Transform API response
        # -----------------------------------------
        raw_df = transform_history(
            coin_id,
            raw_response,
        )

        if raw_df is None or raw_df.empty:
            raise RuntimeError(
                f"No historical data returned for {coin_id}"
            )

        # -----------------------------------------
        # 3. Store RAW history
        # -----------------------------------------
        raw_rows = upsert_raw_history(
            raw_df
        )

        # -----------------------------------------
        # 4. Clean / normalize history
        # -----------------------------------------
        clean_df = clean_history_dataframe(
            raw_df
        )

        if clean_df is None or clean_df.empty:
            raise RuntimeError(
                f"No cleaned data produced for {coin_id}"
            )

        # -----------------------------------------
        # 5. Store CLEAN hourly history
        # -----------------------------------------
        hourly_rows = upsert_hourly_history(
            clean_df
        )

        print(
            f"{coin_id}: "
            f"{raw_rows} raw rows upserted, "
            f"{hourly_rows} hourly rows upserted."
        )

        return True

    except Exception as exc:
        print(
            f"{coin_id}: FAILED - {exc}"
        )

        return False


def run_daily_pipeline():
    """
    Main daily ingestion pipeline.

    1. Fetch current market snapshots.
    2. Save snapshots to PostgreSQL.
    3. Fetch recent history for each coin.
    4. Save raw history.
    5. Clean history.
    6. Save cleaned hourly history.
    """

    print("Starting daily market pipeline...")

    # =====================================================
    # CURRENT MARKET SNAPSHOT
    # =====================================================

    snapshot_rows = process_market_snapshots()

    # =====================================================
    # HISTORICAL MARKET DATA
    # =====================================================

    successful_coins = []
    failed_coins = []

    for coin_id in COINS:

        success = process_coin(
            coin_id
        )

        if success:
            successful_coins.append(
                coin_id
            )
        else:
            failed_coins.append(
                coin_id
            )

    # =====================================================
    # SUMMARY
    # =====================================================

    print(
        "\n========== PIPELINE SUMMARY =========="
    )

    print(
        f"Market snapshot rows: {snapshot_rows}"
    )

    print(
        f"Successful coins: {len(successful_coins)}"
    )

    if successful_coins:
        print(
            "Successful coin list: "
            + ", ".join(successful_coins)
        )

    print(
        f"Failed coins: {len(failed_coins)}"
    )

    if failed_coins:
        print(
            "Failed coin list: "
            + ", ".join(failed_coins)
        )

    if failed_coins:
        print(
            "Daily market pipeline completed with failures."
        )
    else:
        print(
            "Daily market pipeline completed successfully."
        )

    return {
        "snapshot_rows": snapshot_rows,
        "successful_coins": successful_coins,
        "failed_coins": failed_coins,
    }


if __name__ == "__main__":
    run_daily_pipeline()