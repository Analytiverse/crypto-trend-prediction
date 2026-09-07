from config import COINS
from src.database.repository import (
    upsert_hourly_history,
    upsert_raw_history,
)
from src.ingestion.fetch_history import (
    fetch_coin_history,
    transform_history,
)
from src.processing.clean_history import clean_history_dataframe


RECENT_HISTORY_DAYS = 2


def process_coin(coin_id):
    """
    Fetch recent CoinGecko history for one coin,
    store raw observations, clean them, and store hourly data.
    """
    print(f"\nProcessing {coin_id}...")

    try:
        data = fetch_coin_history(
            coin_id=coin_id,
            days=RECENT_HISTORY_DAYS,
        )

        if data is None:
            print(f"{coin_id}: API request failed.")

            return {
                "coin_id": coin_id,
                "success": False,
                "raw_rows": 0,
                "hourly_rows": 0,
                "error": "API request failed",
            }

        raw_df = transform_history(
            coin_id=coin_id,
            data=data,
        )

        if raw_df.empty:
            print(f"{coin_id}: no usable observations returned.")

            return {
                "coin_id": coin_id,
                "success": False,
                "raw_rows": 0,
                "hourly_rows": 0,
                "error": "No usable observations returned",
            }

        raw_count = upsert_raw_history(raw_df)

        clean_df = clean_history_dataframe(raw_df)

        if clean_df.empty:
            print(f"{coin_id}: no rows remained after cleaning.")

            return {
                "coin_id": coin_id,
                "success": False,
                "raw_rows": raw_count,
                "hourly_rows": 0,
                "error": "No rows remained after cleaning",
            }

        hourly_count = upsert_hourly_history(clean_df)

        print(
            f"{coin_id}: "
            f"{raw_count} raw rows upserted, "
            f"{hourly_count} hourly rows upserted."
        )

        return {
            "coin_id": coin_id,
            "success": True,
            "raw_rows": raw_count,
            "hourly_rows": hourly_count,
            "error": None,
        }

    except Exception as exc:
        print(f"{coin_id}: pipeline failed: {exc}")

        return {
            "coin_id": coin_id,
            "success": False,
            "raw_rows": 0,
            "hourly_rows": 0,
            "error": str(exc),
        }


def run_daily_pipeline():
    """
    Run the daily market ingestion pipeline.

    Returns a dictionary so the function can be called
    from both the command line and an HTTP endpoint.
    """
    successful_coins = []
    failed_coins = []
    coin_results = []

    print("Starting daily market pipeline...")

    for coin_id in COINS:
        result = process_coin(coin_id)

        coin_results.append(result)

        if result["success"]:
            successful_coins.append(coin_id)
        else:
            failed_coins.append(coin_id)

    print("\n========== PIPELINE SUMMARY ==========")

    print(f"Successful coins: {len(successful_coins)}")

    if successful_coins:
        print(
            "Successful coin list:",
            ", ".join(successful_coins),
        )

    print(f"Failed coins: {len(failed_coins)}")

    if failed_coins:
        print(
            "Failed coin list:",
            ", ".join(failed_coins),
        )

    result = {
        "status": (
            "success"
            if not failed_coins
            else "failed"
        ),
        "successful_coins": successful_coins,
        "failed_coins": failed_coins,
        "coin_results": coin_results,
    }

    if failed_coins:
        raise RuntimeError(
            f"Pipeline completed with "
            f"{len(failed_coins)} failed coin(s)."
        )

    print("Daily market pipeline completed successfully.")

    return result


def main():
    run_daily_pipeline()


if __name__ == "__main__":
    main()