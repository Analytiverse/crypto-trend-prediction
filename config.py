import os
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()



# =========================
# CoinGecko configuration
# =========================

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

if not COINGECKO_API_KEY:
    raise ValueError("COINGECKO_API_KEY is not set in the environment.")

# Backward-compatible alias used by the API client
API_KEY = COINGECKO_API_KEY

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

# =========================
# Database configuration
# =========================

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment.")


# =========================
# Project paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


RAW_HISTORY_FILE = RAW_DATA_DIR / "market_history_raw.csv"

RAW_SNAPSHOT_FILE = RAW_DATA_DIR / "market_snapshot.csv"

PROCESSED_HISTORY_FILE = PROCESSED_DATA_DIR / "market_hourly.csv"
