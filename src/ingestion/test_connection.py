from src.api.coingecko_client import CoinGeckoClient


def main():
    client = CoinGeckoClient()

    print(
        "Testing CoinGecko connection..."
    )

    data = client.get(
        endpoint="/ping"
    )

    if data is None:
        print(
            "CoinGecko connection failed."
        )

        return

    print(
        "CoinGecko connection successful."
    )

    print(
        "Response:",
        data,
    )


if __name__ == "__main__":
    main()