import os
from datetime import datetime, timezone

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

OUTPUT_FILE = "data/raw/market_snapshots.csv"


def fetch_market_data():
    url = f"{BASE_URL}/coins/markets"

    headers = {
        "x-cg-demo-api-key": API_KEY
    }

    params = {
        "vs_currency": "usd",
        "ids": ",".join(COINS),
        "price_change_percentage": "1h,24h,7d",
        "sparkline": "false",
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def transform_market_data(data):
    collected_at = datetime.now(timezone.utc).isoformat()

    rows = []

    for coin in data:
        rows.append(
            {
                "coin_id": coin["id"],
                "symbol": coin["symbol"].upper(),
                "name": coin["name"],
                "current_price": coin["current_price"],
                "market_cap": coin["market_cap"],
                "market_cap_rank": coin["market_cap_rank"],
                "total_volume": coin["total_volume"],
                "high_24h": coin["high_24h"],
                "low_24h": coin["low_24h"],
                "price_change_24h": coin["price_change_24h"],
                "price_change_percentage_24h":
                    coin["price_change_percentage_24h"],
                "price_change_percentage_1h":
                    coin.get("price_change_percentage_1h_in_currency"),
                "price_change_percentage_7d":
                    coin.get("price_change_percentage_7d_in_currency"),
                "circulating_supply": coin["circulating_supply"],
                "last_updated": coin["last_updated"],
                "collected_at": collected_at,
            }
        )

    return pd.DataFrame(rows)


def save_without_duplicates(df):
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)

        combined_df = pd.concat(
            [existing_df, df],
            ignore_index=True
        )
    else:
        combined_df = df

    combined_df = combined_df.drop_duplicates(
        subset=["coin_id", "last_updated"],
        keep="last"
    )

    combined_df = combined_df.sort_values(
        ["coin_id", "last_updated"]
    )

    combined_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Fetched rows: {len(df)}")
    print(f"Total unique rows: {len(combined_df)}")


def main():
    data = fetch_market_data()

    df = transform_market_data(data)

    print(df)

    save_without_duplicates(df)


if __name__ == "__main__":
    main()