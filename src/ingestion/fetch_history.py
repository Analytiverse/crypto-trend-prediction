import os

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("COINGECKO_API_KEY")

BASE_URL = "https://api.coingecko.com/api/v3"

COINS = [
    "bitcoin",
    "ethereum",
    "solana",
    "ripple",
    "cardano",
]

OUTPUT_FILE = "data/raw/market_history.csv"


def fetch_coin_history(coin_id, days=90):
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"

    headers = {
        "x-cg-demo-api-key": API_KEY
    }

    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "hourly",
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def transform_history(coin_id, data):
    price_df = pd.DataFrame(
        data["prices"],
        columns=["timestamp_ms", "price"]
    )

    market_cap_df = pd.DataFrame(
        data["market_caps"],
        columns=["timestamp_ms", "market_cap"]
    )

    volume_df = pd.DataFrame(
        data["total_volumes"],
        columns=["timestamp_ms", "total_volume"]
    )

    df = price_df.merge(
        market_cap_df,
        on="timestamp_ms",
        how="outer"
    )

    df = df.merge(
        volume_df,
        on="timestamp_ms",
        how="outer"
    )

    df["coin_id"] = coin_id

    df["timestamp"] = pd.to_datetime(
        df["timestamp_ms"],
        unit="ms",
        utc=True
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


def save_without_duplicates(new_df):
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)

        combined_df = pd.concat(
            [existing_df, new_df],
            ignore_index=True
        )
    else:
        combined_df = new_df

    combined_df = combined_df.drop_duplicates(
        subset=["coin_id", "timestamp_ms"],
        keep="last"
    )

    combined_df = combined_df.sort_values(
        ["coin_id", "timestamp_ms"]
    )

    combined_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Fetched rows: {len(new_df)}")
    print(f"Total unique rows: {len(combined_df)}")


def main():
    all_data = []

    for coin in COINS:
        print(f"Fetching historical data for {coin}...")

        data = fetch_coin_history(
            coin_id=coin,
            days=90
        )

        df = transform_history(
            coin_id=coin,
            data=data
        )

        all_data.append(df)

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    save_without_duplicates(final_df)

    print("\nSample data:")
    print(final_df.head())


if __name__ == "__main__":
    main()