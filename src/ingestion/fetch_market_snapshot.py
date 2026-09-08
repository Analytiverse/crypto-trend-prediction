from datetime import datetime, timezone

import pandas as pd

from src.api.coingecko_client import CoinGeckoClient
from src.config import COINS, VS_CURRENCY


client = CoinGeckoClient()


def fetch_market_data():
    """
    Fetch current market data for configured coins
    from CoinGecko /coins/markets.
    """

    print("Fetching current market data...")

    coin_ids = ",".join(COINS)

    params = {
        "vs_currency": VS_CURRENCY,
        "ids": coin_ids,
        "order": "market_cap_desc",
        "per_page": len(COINS),
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d",
    }

    data = client.get(
        "/coins/markets",
        params=params,
    )

    if not isinstance(data, list):
        raise RuntimeError(
            "Unexpected response from CoinGecko /coins/markets"
        )

    return data


def transform_market_data(data):
    """
    Convert CoinGecko market response
    into a normalized DataFrame.
    """

    collected_at = datetime.now(
        timezone.utc
    )

    rows = []

    for coin in data:

        rows.append(
            {
                "coin_id": coin.get("id"),
                "symbol": (
                    coin.get("symbol") or ""
                ).upper(),
                "name": coin.get("name"),
                "current_price": coin.get(
                    "current_price"
                ),
                "market_cap": coin.get(
                    "market_cap"
                ),
                "market_cap_rank": coin.get(
                    "market_cap_rank"
                ),
                "total_volume": coin.get(
                    "total_volume"
                ),
                "high_24h": coin.get(
                    "high_24h"
                ),
                "low_24h": coin.get(
                    "low_24h"
                ),
                "price_change_24h": coin.get(
                    "price_change_24h"
                ),
                "price_change_percentage_24h":
                    coin.get(
                        "price_change_percentage_24h"
                    ),
                "price_change_percentage_1h":
                    coin.get(
                        "price_change_percentage_1h_in_currency"
                    ),
                "price_change_percentage_7d":
                    coin.get(
                        "price_change_percentage_7d_in_currency"
                    ),
                "circulating_supply":
                    coin.get(
                        "circulating_supply"
                    ),
                "total_supply":
                    coin.get(
                        "total_supply"
                    ),
                "max_supply":
                    coin.get(
                        "max_supply"
                    ),
                "last_updated":
                    coin.get(
                        "last_updated"
                    ),
                "collected_at":
                    collected_at,
            }
        )

    df = pd.DataFrame(rows)

    return df


def main():
    """
    Test current market fetch locally.
    Does not save CSV.
    """

    try:

        raw_data = fetch_market_data()

        df = transform_market_data(
            raw_data
        )

        print()
        print(
            f"Fetched {len(df)} coin records"
        )
        print()

        if df.empty:
            print(
                "No market data returned."
            )
            return

        display_columns = [
            "coin_id",
            "symbol",
            "current_price",
            "market_cap",
            "total_volume",
            "last_updated",
        ]

        print(
            df[
                display_columns
            ].to_string(
                index=False
            )
        )

    except Exception as exc:

        print(
            f"Market fetch failed: {exc}"
        )


if __name__ == "__main__":
    main()