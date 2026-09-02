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

OUTPUT_FILE = "data/raw/coins.csv"


def fetch_coins():
    url = f"{BASE_URL}/coins/markets"

    headers = {
        "x-cg-demo-api-key": API_KEY
    }

    params = {
        "vs_currency": "usd",
        "ids": ",".join(COINS),
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


def transform_coins(data):
    rows = []

    for coin in data:
        rows.append(
            {
                "coin_id": coin["id"],
                "symbol": coin["symbol"].upper(),
                "name": coin["name"],
            }
        )

    df = pd.DataFrame(rows)

    # coin_id must be unique in our master table
    df = df.drop_duplicates(
        subset=["coin_id"]
    )

    return df.sort_values("coin_id")


def main():
    data = fetch_coins()

    df = transform_coins(data)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nCoin master data:")
    print(df)

    print(f"\nSaved {len(df)} coins to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()