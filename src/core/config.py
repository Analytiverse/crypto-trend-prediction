import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# Environment
# ============================================================

# Repository root:
# crypto-trend-prediction/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Explicitly load the repository-level .env file so configuration
# behaves consistently regardless of the current working directory.
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE)


# ============================================================
# CoinGecko
# ============================================================

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

if not COINGECKO_API_KEY:
    raise ValueError("COINGECKO_API_KEY is not set in the environment.")

# Backward-compatible name currently used by the CoinGecko client.
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


# ============================================================
# Database
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment.")


# ============================================================
# Data paths
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_HISTORY_FILE = RAW_DATA_DIR / "market_history_raw.csv"
RAW_SNAPSHOT_FILE = RAW_DATA_DIR / "market_snapshot.csv"
PROCESSED_HISTORY_FILE = PROCESSED_DATA_DIR / "market_hourly.csv"
