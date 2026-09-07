import os
from datetime import datetime, timezone

import pandas as pd

from src.api.coingecko_client import CoinGeckoClient
from config import (
    COINS,
    RAW_SNAPSHOT_FILE,
    VS_CURRENCY,
)


client = CoinGeckoClient()


def fetch_market_data():
    params = {
        "vs_currency": VS_CURRENCY,
        "ids": ",".join(COINS),
        "price_change_percentage": "1h,24h,7d",
        "sparkline": "false",
    }

    return client.get(
        endpoint="/coins/markets",
        params=params,
    )


def transform_market_data(
    data,
):
    collected_at = (
        datetime
        .now(timezone.utc)
        .isoformat()
    )

    rows = []

    for coin in data:
        try:
            row = {
                "coin_id": coin["id"],
                "symbol": coin["symbol"].upper(),
                "name": coin["name"],
                "current_price": coin["current_price"],
                "market_cap": coin["market_cap"],
                "market_cap_rank": coin["market_cap_rank"],
                "total_volume": coin["total_volume"],
                "high_24h": coin["high_24h"],
                "low_24h": coin["low_24h"],
                "price_change_24h":
                    coin["price_change_24h"],
                "price_change_percentage_24h":
                    coin[
                        "price_change_percentage_24h"
                    ],
                "price_change_percentage_1h":
                    coin.get(
                        "price_change_percentage_1h_in_currency"
                    ),
                "price_change_percentage_7d":
                    coin.get(
                        "price_change_percentage_7d_in_currency"
                    ),
                "circulating_supply":
                    coin["circulating_supply"],
                "last_updated":
                    coin["last_updated"],
                "collected_at":
                    collected_at,
            }

            rows.append(row)

        except KeyError as exc:
            print(
                f"Skipping malformed coin record. "
                f"Missing field: {exc}"
            )

    return pd.DataFrame(rows)


def save_without_duplicates(
    new_df,
):
    if os.path.exists(RAW_SNAPSHOT_FILE):
        existing_df = pd.read_csv(
            RAW_SNAPSHOT_FILE
        )

        combined_df = pd.concat(
            [
                existing_df,
                new_df,
            ],
            ignore_index=True,
        )

    else:
        combined_df = new_df.copy()

    combined_df = (
        combined_df
        .drop_duplicates(
            subset=[
                "coin_id",
                "last_updated",
            ],
            keep="last",
        )
        .sort_values(
            [
                "coin_id",
                "last_updated",
            ]
        )
        .reset_index(drop=True)
    )

    combined_df.to_csv(
        RAW_SNAPSHOT_FILE,
        index=False,
    )

    print(
        f"Fetched rows in this run: "
        f"{len(new_df)}"
    )

    print(
        f"Total unique rows stored: "
        f"{len(combined_df)}"
    )


def main():
    print(
        "Fetching current market data..."
    )

    try:
        data = fetch_market_data()

        if data is None:
            print(
                "Snapshot ingestion failed "
                "because the API request did not succeed."
            )

            return

        if not isinstance(data, list):
            print(
                "Unexpected response format "
                "from CoinGecko."
            )

            return

        df = transform_market_data(
            data
        )

        if df.empty:
            print(
                "No valid market snapshot "
                "records were returned."
            )

            return

        try:
            save_without_duplicates(
                df
            )

        except PermissionError:
            print(
                f"Could not write to "
                f"{RAW_SNAPSHOT_FILE}."
            )

            print(
                "Make sure the CSV is not "
                "currently open in Excel."
            )

        except Exception as exc:
            print(
                f"Failed while saving "
                f"snapshot data: {exc}"
            )

    except Exception as exc:
        print(
            f"Snapshot ingestion failed: "
            f"{exc}"
        )


if __name__ == "__main__":
    main()