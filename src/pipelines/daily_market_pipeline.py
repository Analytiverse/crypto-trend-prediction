from src.config import COINS

from src.database.repository import (
    upsert_hourly_history,
    upsert_raw_history,
    upsert_market_snapshots,
)

from src.database.repair_gaps import (
    repair_detected_gaps,
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


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Fetch the most recent 2 days of hourly history.
#
# This provides overlap between daily runs and helps recover
# short ingestion failures automatically.
RECENT_HISTORY_DAYS = 2


# After normal ingestion, check this many recent days
# for missing hourly observations.
#
# This gives us a larger safety window than the normal
# 48-hour fetch.
GAP_REPAIR_LOOKBACK_DAYS = 7


def process_market_snapshots():
    """
    Fetch latest /coins/markets data and store it
    in the market_snapshots table.
    """

    print(
        "\nFetching current market snapshots..."
    )

    try:

        raw_data = fetch_market_data()

        if not raw_data:
            print(
                "No market snapshot data returned."
            )

            return 0

        snapshot_df = (
            transform_market_data(
                raw_data
            )
        )

        if (
            snapshot_df is None
            or snapshot_df.empty
        ):
            print(
                "Snapshot DataFrame is empty."
            )

            return 0

        print(
            f"Fetched "
            f"{len(snapshot_df)} "
            f"current market snapshot rows."
        )

        rows_upserted = (
            upsert_market_snapshots(
                snapshot_df
            )
        )

        print(
            f"{rows_upserted} "
            f"market snapshot rows upserted."
        )

        return rows_upserted

    except Exception as exc:

        print(
            f"Market snapshot processing "
            f"failed: {exc}"
        )

        return 0


def process_coin(
    coin_id,
):
    """
    Fetch recent historical data for one coin,
    store raw data, clean it, and store hourly data.
    """

    print(
        f"\nProcessing {coin_id}..."
    )

    try:

        # -----------------------------------------
        # 1. Fetch recent CoinGecko history
        # -----------------------------------------

        raw_response = (
            fetch_coin_history(
                coin_id,
                days=RECENT_HISTORY_DAYS,
            )
        )

        if not raw_response:
            raise RuntimeError(
                f"No API response returned "
                f"for {coin_id}"
            )

        # -----------------------------------------
        # 2. Transform API response
        # -----------------------------------------

        raw_df = transform_history(
            coin_id,
            raw_response,
        )

        if (
            raw_df is None
            or raw_df.empty
        ):
            raise RuntimeError(
                f"No historical data "
                f"returned for {coin_id}"
            )

        # -----------------------------------------
        # 3. Store raw history
        # -----------------------------------------

        raw_rows = (
            upsert_raw_history(
                raw_df
            )
        )

        # -----------------------------------------
        # 4. Clean / normalize history
        # -----------------------------------------

        clean_df = (
            clean_history_dataframe(
                raw_df
            )
        )

        if (
            clean_df is None
            or clean_df.empty
        ):
            raise RuntimeError(
                f"No cleaned data "
                f"produced for {coin_id}"
            )

        # -----------------------------------------
        # 5. Store clean hourly history
        # -----------------------------------------

        hourly_rows = (
            upsert_hourly_history(
                clean_df
            )
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

    Steps:
    1. Fetch current market snapshots.
    2. Save snapshots to PostgreSQL.
    3. Fetch the previous 48 hours for each coin.
    4. Save raw historical data.
    5. Clean / normalize historical data.
    6. Save cleaned hourly data.
    7. Check recent history for hourly gaps.
    8. Attempt to repair detected gaps.
    """

    print(
        "Starting daily market pipeline..."
    )

    # =====================================================
    # CURRENT MARKET SNAPSHOT
    # =====================================================

    snapshot_rows = (
        process_market_snapshots()
    )

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
    # AUTOMATIC GAP CHECK / REPAIR
    # =====================================================

    print(
        "\nChecking recent hourly "
        "history for gaps..."
    )

    try:

        gap_repair_summary = (
            repair_detected_gaps(
                lookback_days=
                    GAP_REPAIR_LOOKBACK_DAYS
            )
        )

    except Exception as exc:

        print(
            f"Automatic gap repair failed: "
            f"{exc}"
        )

        gap_repair_summary = {
            "gaps_before": None,
            "successful_repairs": 0,
            "failed_repairs": 0,
            "gaps_after": None,
        }

    # =====================================================
    # SUMMARY
    # =====================================================

    print(
        "\n========== PIPELINE SUMMARY =========="
    )

    print(
        f"Market snapshot rows: "
        f"{snapshot_rows}"
    )

    print(
        f"Successful coins: "
        f"{len(successful_coins)}"
    )

    if successful_coins:

        print(
            "Successful coin list: "
            + ", ".join(
                successful_coins
            )
        )

    print(
        f"Failed coins: "
        f"{len(failed_coins)}"
    )

    if failed_coins:

        print(
            "Failed coin list: "
            + ", ".join(
                failed_coins
            )
        )

    print(
        f"Gaps detected before repair: "
        f"{gap_repair_summary['gaps_before']}"
    )

    print(
        f"Gaps remaining after repair: "
        f"{gap_repair_summary['gaps_after']}"
    )

    if failed_coins:

        print(
            "Daily market pipeline "
            "completed with failures."
        )

    else:

        print(
            "Daily market pipeline "
            "completed successfully."
        )

    return {
        "snapshot_rows":
            snapshot_rows,

        "successful_coins":
            successful_coins,

        "failed_coins":
            failed_coins,

        "gap_repair":
            gap_repair_summary,
    }


if __name__ == "__main__":
    run_daily_pipeline()