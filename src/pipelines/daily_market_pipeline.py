"""
AlphaPulse Daily Market Pipeline

Daily production workflow:

1. Fetch current CoinGecko market snapshots.
2. Store snapshots in PostgreSQL.
3. Fetch recent historical data for every supported coin.
4. Store raw history.
5. Clean and normalize hourly history.
6. Store cleaned hourly history.
7. Detect and repair recent gaps.
8. Generate production predictions.
9. Persist latest predictions to PostgreSQL.

Important:
    This pipeline performs inference only.
    It does NOT retrain production models.
"""

from src.core.config import COINS

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

from src.prediction.batch_predictor import (
    run_batch_predictions,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Fetch the most recent two days of hourly history.
#
# The overlap allows normal daily runs to recover from
# short ingestion interruptions.
RECENT_HISTORY_DAYS = 2


# Check a wider recent window for missing hourly observations
# after normal ingestion.
GAP_REPAIR_LOOKBACK_DAYS = 7


# ============================================================
# MARKET SNAPSHOTS
# ============================================================

def process_market_snapshots():
    """
    Fetch latest /coins/markets data and store it
    in the market_snapshots table.
    """

    print(
        "\nFetching current market snapshots..."
    )

    try:

        raw_data = (
            fetch_market_data()
        )

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
            "Market snapshot processing "
            f"failed: {exc}"
        )

        return 0


# ============================================================
# SINGLE COIN INGESTION
# ============================================================

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

        # ----------------------------------------------------
        # 1. Fetch recent CoinGecko history
        # ----------------------------------------------------

        raw_response = (
            fetch_coin_history(
                coin_id,
                days=RECENT_HISTORY_DAYS,
            )
        )

        if not raw_response:

            raise RuntimeError(
                "No API response returned "
                f"for {coin_id}"
            )

        # ----------------------------------------------------
        # 2. Transform API response
        # ----------------------------------------------------

        raw_df = (
            transform_history(
                coin_id,
                raw_response,
            )
        )

        if (
            raw_df is None
            or raw_df.empty
        ):

            raise RuntimeError(
                "No historical data "
                f"returned for {coin_id}"
            )

        # ----------------------------------------------------
        # 3. Store raw history
        # ----------------------------------------------------

        raw_rows = (
            upsert_raw_history(
                raw_df
            )
        )

        # ----------------------------------------------------
        # 4. Clean / normalize history
        # ----------------------------------------------------

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
                "No cleaned data "
                f"produced for {coin_id}"
            )

        # ----------------------------------------------------
        # 5. Store cleaned hourly history
        # ----------------------------------------------------

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


# ============================================================
# PRODUCTION PREDICTIONS
# ============================================================

def run_daily_predictions():
    """
    Generate and persist production predictions.

    Uses the already-trained production models.

    No model retraining occurs here.
    """

    print()
    print(
        "Starting daily production predictions..."
    )

    try:

        prediction_summary = (
            run_batch_predictions(
                persist=True,
            )
        )

        successful = (
            prediction_summary.get(
                "successful_predictions",
                0,
            )
        )

        failed = (
            prediction_summary.get(
                "failed_predictions",
                0,
            )
        )

        stored = (
            prediction_summary.get(
                "stored_predictions",
                0,
            )
        )

        timestamp = (
            prediction_summary.get(
                "prediction_timestamp"
            )
        )

        timestamp_consistent = (
            prediction_summary.get(
                "timestamp_consistent",
                False,
            )
        )

        print()
        print(
            "Daily prediction summary:"
        )

        print(
            f"Successful predictions: "
            f"{successful}"
        )

        print(
            f"Failed predictions: "
            f"{failed}"
        )

        print(
            f"Stored predictions: "
            f"{stored}"
        )

        print(
            f"Prediction timestamp: "
            f"{timestamp}"
        )

        print(
            "Timestamp consistent: "
            f"{timestamp_consistent}"
        )

        return prediction_summary

    except Exception as exc:

        print(
            "Daily prediction generation "
            f"failed: {exc}"
        )

        return {
            "status":
                "FAILED",

            "successful_predictions":
                0,

            "failed_predictions":
                15,

            "stored_predictions":
                0,

            "prediction_timestamp":
                None,

            "timestamp_consistent":
                False,

            "error":
                str(exc),
        }


# ============================================================
# DAILY PIPELINE
# ============================================================

def run_daily_pipeline():
    """
    Run the complete AlphaPulse daily production pipeline.

    Steps:
    1. Fetch current market snapshots.
    2. Save snapshots to PostgreSQL.
    3. Fetch previous 48 hours for every coin.
    4. Save raw historical data.
    5. Clean and normalize historical data.
    6. Save cleaned hourly data.
    7. Detect and repair recent gaps.
    8. Generate production predictions if ingestion succeeded.
    9. Persist successful predictions to PostgreSQL.

    Production models are loaded for inference only.
    They are NOT retrained here.
    """

    print()
    print("=" * 70)
    print(
        "ALPHAPULSE DAILY MARKET PIPELINE"
    )
    print("=" * 70)

    # ========================================================
    # CURRENT MARKET SNAPSHOT
    # ========================================================

    snapshot_rows = (
        process_market_snapshots()
    )

    # ========================================================
    # HISTORICAL MARKET DATA
    # ========================================================

    successful_coins = []

    failed_coins = []

    for coin_id in COINS:

        success = (
            process_coin(
                coin_id
            )
        )

        if success:

            successful_coins.append(
                coin_id
            )

        else:

            failed_coins.append(
                coin_id
            )

    # ========================================================
    # AUTOMATIC GAP CHECK / REPAIR
    # ========================================================

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
            "Automatic gap repair failed: "
            f"{exc}"
        )

        gap_repair_summary = {

            "gaps_before":
                None,

            "successful_repairs":
                0,

            "failed_repairs":
                0,

            "gaps_after":
                None,
        }

    # ========================================================
    # PRODUCTION PREDICTIONS
    # ========================================================

    prediction_summary = None

    if failed_coins:

        print()
        print("=" * 70)
        print(
            "PREDICTIONS SKIPPED"
        )
        print("=" * 70)

        print(
            "One or more coins failed ingestion."
        )

        print(
            "Fresh predictions will not be generated "
            "from a partially updated dataset."
        )

    else:

        prediction_summary = (
            run_daily_predictions()
        )

    # ========================================================
    # PIPELINE SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "PIPELINE SUMMARY"
    )
    print("=" * 70)

    print(
        "Market snapshot rows: "
        f"{snapshot_rows}"
    )

    print(
        "Successful coins: "
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
        "Failed coins: "
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
        "Gaps detected before repair: "
        f"{gap_repair_summary['gaps_before']}"
    )

    print(
        "Gaps remaining after repair: "
        f"{gap_repair_summary['gaps_after']}"
    )

    if prediction_summary:

        print(
            "Successful predictions: "
            f"{prediction_summary.get('successful_predictions', 0)}"
        )

        print(
            "Failed predictions: "
            f"{prediction_summary.get('failed_predictions', 0)}"
        )

        print(
            "Stored predictions: "
            f"{prediction_summary.get('stored_predictions', 0)}"
        )

        print(
            "Prediction timestamp: "
            f"{prediction_summary.get('prediction_timestamp')}"
        )

    else:

        print(
            "Predictions: SKIPPED"
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    prediction_failed = (
        prediction_summary is not None
        and prediction_summary.get(
            "failed_predictions",
            0,
        ) > 0
    )

    if (
        failed_coins
        or prediction_failed
    ):

        print(
            "Daily market pipeline "
            "completed with failures."
        )

    else:

        print(
            "Daily market pipeline "
            "completed successfully."
        )

    # ========================================================
    # API RESPONSE
    # ========================================================

    return {

        "snapshot_rows":
            snapshot_rows,

        "successful_coins":
            successful_coins,

        "failed_coins":
            failed_coins,

        "gap_repair":
            gap_repair_summary,

        "predictions":
            prediction_summary,
    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    run_daily_pipeline()