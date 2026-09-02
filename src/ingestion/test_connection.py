import os

import requests
from dotenv import load_dotenv


# Load variables from the .env file
load_dotenv()

# Read our CoinGecko API key
API_KEY = os.getenv("COINGECKO_API_KEY")

BASE_URL = "https://api.coingecko.com/api/v3"


def test_connection():
    if not API_KEY:
        raise ValueError(
            "COINGECKO_API_KEY was not found in the .env file."
        )

    headers = {
        "x-cg-demo-api-key": API_KEY
    }

    response = requests.get(
        f"{BASE_URL}/ping",
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    print("CoinGecko connection successful!")
    print("Response:", response.json())


if __name__ == "__main__":
    test_connection()