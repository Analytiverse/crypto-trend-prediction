import os

import pandas as pd

from src.api.coingecko_client import CoinGeckoClient
from config import (
    COINS,
    RAW_HISTORY_FILE,
    VS_CURRENCY,
)


client = CoinGeckoClient()


def fetch_coin_history(
    coin_id,
    days=90,
):
    params = {
        "vs_currency": VS_CURRENCY,
        "days": days,
        "interval": "hourly",
    }

    return client.get(
        endpoint=f"/coins/{coin_id}/market_chart",
        params=params,
    )


def transform_history(
    coin_id,
    data,
):
    required_keys = [
        "prices",
        "market_caps",
        "total_volumes",
    ]

    for key in required_keys:
        if key not in data:
            raise ValueError(
                f"Missing expected API field: {key}"
            )

    price_df = pd.DataFrame(
        data["prices"],
        columns=[
            "timestamp_ms",
            "price",
        ],
    )

    market_cap_df = pd.DataFrame(
        data["market_caps"],
        columns=[
            "timestamp_ms",
            "market_cap",
        ],
    )

    volume_df = pd.DataFrame(
        data["total_volumes"],
        columns=[
            "timestamp_ms",
            "total_volume",
        ],
    )

    df = price_df.merge(
        market_cap_df,
        on="timestamp_ms",
        how="outer",
    )

    df = df.merge(
        volume_df,
        on="timestamp_ms",
        how="outer",
    )

    df["coin_id"] = coin_id

    df["timestamp"] = pd.to_datetime(
        df["timestamp_ms"],
        unit="ms",
        utc=True,
    )

    return df[
        [
            "coin_id",
            "timestamp_ms",
            "timestamp",
            "price",
            "market_cap",
            "total_volume",
        ]
    ]


def save_without_duplicates(
    new_df,
):
    if os.path.exists(RAW_HISTORY_FILE):
        existing_df = pd.read_csv(
            RAW_HISTORY_FILE
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
                "timestamp_ms",
            ],
            keep="last",
        )
        .sort_values(
            [
                "coin_id",
                "timestamp_ms",
            ]
        )
        .reset_index(drop=True)
    )

    combined_df.to_csv(
        RAW_HISTORY_FILE,
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


def process_coin(
    coin_id,
):
    print(
        f"\nFetching historical data "
        f"for {coin_id}..."
    )

    try:
        data = fetch_coin_history(
            coin_id=coin_id,
            days=90,
        )

        if data is None:
            print(
                f"Skipping {coin_id}: "
                f"API request failed."
            )

            return None

        df = transform_history(
            coin_id=coin_id,
            data=data,
        )

        if df.empty:
            print(
                f"Skipping {coin_id}: "
                f"API returned no usable rows."
            )

            return None

        print(
            f"{coin_id}: "
            f"{len(df)} rows fetched successfully."
        )

        return df

    except Exception as exc:
        print(
            f"Failed to process "
            f"{coin_id}: {exc}"
        )

        return None


def main():
    successful_data = []

    successful_coins = []

    failed_coins = []

    for coin in COINS:
        df = process_coin(
            coin_id=coin
        )

        if df is not None:
            successful_data.append(df)

            successful_coins.append(coin)

        else:
            failed_coins.append(coin)

    if successful_data:
        final_df = pd.concat(
            successful_data,
            ignore_index=True,
        )

        try:
            save_without_duplicates(
                final_df
            )

        except PermissionError:
            print(
                "\nCould not write to "
                f"{RAW_HISTORY_FILE}."
            )

            print(
                "Make sure the CSV is not "
                "currently open in Excel."
            )

        except Exception as exc:
            print(
                f"\nFailed while saving "
                f"historical data: {exc}"
            )

    else:
        print(
            "\nNo historical data was "
            "fetched successfully."
        )

    print(
        "\n========== INGESTION SUMMARY =========="
    )

    print(
        f"Successful coins: "
        f"{len(successful_coins)}"
    )

    if successful_coins:
        print(
            "Successful coin list:",
            ", ".join(successful_coins),
        )

    print(
        f"Failed coins: "
        f"{len(failed_coins)}"
    )

    if failed_coins:
        print(
            "Failed coin list:",
            ", ".join(failed_coins),
        )


if __name__ == "__main__":
    main()