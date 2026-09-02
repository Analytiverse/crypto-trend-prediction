import os
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

VS_CURRENCY = "usd"

REQUEST_TIMEOUT = 30

MAX_RETRIES = 3

RETRY_BACKOFF_SECONDS = 2

RAW_HISTORY_FILE = "data/raw/market_history.csv"

RAW_SNAPSHOT_FILE = "data/raw/market_snapshots.csv"

PROCESSED_HISTORY_FILE = "data/processed/market_hourly.csv"