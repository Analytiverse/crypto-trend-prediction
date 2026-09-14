"""
Production feature construction for AlphaPulse.

This module creates exactly the feature set used by the final
machine-learning models.

Important:
- PostgreSQL remains the source of truth.
- No future information is used.
- Historical returns use exact timestamp matching.
- Volume changes use exact timestamp matching.
- Rolling volatility uses historical 1-hour returns only.
- Market context excludes the current coin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DATABASE_URL, COINS


# ============================================================
# MODEL FEATURE CONFIGURATION
# ============================================================

RETURN_HORIZONS = [1, 6, 12, 24]

ML_FEATURES = [
    "return_1h",
    "return_6h",
    "return_12h",
    "return_24h",
    "volatility_24h",
    "volatility_72h",
    "log_volume_change_1h",
    "log_volume_change_6h",
    "log_volume_change_12h",
    "log_volume_change_24h",
    "other_coins_return_1h",
]


# ============================================================
# DATABASE
# ============================================================

def get_sqlalchemy_database_url() -> str:
    """
    Convert the PostgreSQL URL so SQLAlchemy explicitly uses
    the psycopg v3 PostgreSQL driver.
    """
    database_url = DATABASE_URL.strip()

    if database_url.startswith("postgresql+psycopg://"):
        return database_url

    if database_url.startswith("postgresql://"):
        return database_url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )

    if database_url.startswith("postgres://"):
        return database_url.replace(
            "postgres://",
            "postgresql+psycopg://",
            1,
        )

    return database_url


engine = create_engine(
    get_sqlalchemy_database_url(),
    pool_pre_ping=True,
)


# ============================================================
# LOAD ALL MARKET DATA
# ============================================================

def load_all_market_data() -> pd.DataFrame:
    """
    Load the complete cleaned hourly market dataset.

    Used when training production models.
    """

    query = text(
        """
        SELECT
            coin_id,
            timestamp,
            price,
            market_cap,
            total_volume
        FROM market_hourly
        ORDER BY coin_id, timestamp
        """
    )

    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
        )

    if df.empty:
        raise ValueError(
            "market_hourly contains no data."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values(
        ["coin_id", "timestamp"]
    ).reset_index(drop=True)

    return df


# ============================================================
# LOAD RECENT DATA FOR LIVE PREDICTION
# ============================================================

def load_recent_market_data(
    lookback_hours: int = 120,
) -> pd.DataFrame:
    """
    Load recent market history required for live inference.

    The longest feature needs 72 hours of history.
    We use 120 hours so there is additional safety margin.
    """

    if lookback_hours < 96:
        raise ValueError(
            "lookback_hours should be at least 96 hours."
        )

    query = text(
        """
        SELECT
            coin_id,
            timestamp,
            price,
            market_cap,
            total_volume
        FROM market_hourly
        WHERE timestamp >= (
            SELECT
                MAX(timestamp)
                - (:lookback_hours * INTERVAL '1 hour')
            FROM market_hourly
        )
        ORDER BY coin_id, timestamp
        """
    )

    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={
                "lookback_hours": lookback_hours
            },
        )

    if df.empty:
        raise ValueError(
            "No recent market data was found."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values(
        ["coin_id", "timestamp"]
    ).reset_index(drop=True)

    return df


# ============================================================
# VALIDATE RAW INPUT DATA
# ============================================================

def validate_market_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Basic validation before feature construction.
    """

    required_columns = [
        "coin_id",
        "timestamp",
        "price",
        "market_cap",
        "total_volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required market columns: "
            f"{missing_columns}"
        )

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    duplicate_count = df.duplicated(
        subset=["coin_id", "timestamp"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "(coin_id, timestamp) rows."
        )

    if (df["price"] <= 0).any():
        raise ValueError(
            "Invalid price <= 0 detected."
        )

    if (df["market_cap"] <= 0).any():
        raise ValueError(
            "Invalid market_cap <= 0 detected."
        )

    if (df["total_volume"] < 0).any():
        raise ValueError(
            "Invalid negative total_volume detected."
        )

    df = df.sort_values(
        ["coin_id", "timestamp"]
    ).reset_index(drop=True)

    return df


# ============================================================
# HISTORICAL RETURNS
# ============================================================

def create_return_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create historical price returns using exact timestamp
    matching.

    return_h =
        price(t) / price(t-h) - 1

    This matches the EDA/modeling implementation and avoids
    accidentally treating a missing timestamp as a valid
    historical observation.
    """

    df = df.copy()

    for hours in RETURN_HORIZONS:

        previous_prices = df[
            [
                "coin_id",
                "timestamp",
                "price",
            ]
        ].copy()

        previous_prices["timestamp"] = (
            previous_prices["timestamp"]
            + pd.Timedelta(hours=hours)
        )

        previous_prices = previous_prices.rename(
            columns={
                "price":
                    f"price_{hours}h_ago"
            }
        )

        df = df.merge(
            previous_prices,
            on=[
                "coin_id",
                "timestamp",
            ],
            how="left",
        )

        df[f"return_{hours}h"] = (
            df["price"]
            / df[f"price_{hours}h_ago"]
            - 1
        )

    return df


# ============================================================
# ROLLING VOLATILITY
# ============================================================

def create_volatility_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create 24-hour and 72-hour rolling volatility using
    historical 1-hour returns.
    """

    df = df.sort_values(
        ["coin_id", "timestamp"]
    ).copy()

    df["volatility_24h"] = (
        df.groupby(
            "coin_id"
        )["return_1h"]
        .transform(
            lambda values:
                values.rolling(
                    window=24,
                    min_periods=24,
                ).std()
        )
    )

    df["volatility_72h"] = (
        df.groupby(
            "coin_id"
        )["return_1h"]
        .transform(
            lambda values:
                values.rolling(
                    window=72,
                    min_periods=72,
                ).std()
        )
    )

    return df


# ============================================================
# VOLUME FEATURES
# ============================================================

def create_volume_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create timestamp-aligned historical volume values and
    log-volume changes.

    log_volume_change_h =
        log(volume(t) / volume(t-h))
    """

    df = df.copy()

    for hours in RETURN_HORIZONS:

        previous_volume = df[
            [
                "coin_id",
                "timestamp",
                "total_volume",
            ]
        ].copy()

        previous_volume["timestamp"] = (
            previous_volume["timestamp"]
            + pd.Timedelta(hours=hours)
        )

        previous_column = (
            f"volume_previous_{hours}h"
        )

        previous_volume = (
            previous_volume.rename(
                columns={
                    "total_volume":
                        previous_column
                }
            )
        )

        df = df.merge(
            previous_volume,
            on=[
                "coin_id",
                "timestamp",
            ],
            how="left",
        )

        feature_column = (
            f"log_volume_change_{hours}h"
        )

        valid = (
            (df["total_volume"] > 0)
            &
            (df[previous_column] > 0)
        )

        df[feature_column] = np.nan

        df.loc[
            valid,
            feature_column,
        ] = np.log(
            df.loc[
                valid,
                "total_volume",
            ]
            /
            df.loc[
                valid,
                previous_column,
            ]
        )

    return df


# ============================================================
# MARKET CONTEXT
# ============================================================

def create_market_context_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Average 1-hour return of OTHER cryptocurrencies at the
    same timestamp.

    The coin being predicted is explicitly excluded.
    """

    df = df.copy()

    market_stats = (
        df.groupby(
            "timestamp"
        )["return_1h"]
        .agg(
            ["sum", "count"]
        )
        .reset_index()
        .rename(
            columns={
                "sum":
                    "market_return_sum",
                "count":
                    "market_return_count",
            }
        )
    )

    df = df.merge(
        market_stats,
        on="timestamp",
        how="left",
    )

    denominator = (
        df["market_return_count"] - 1
    )

    df["other_coins_return_1h"] = np.where(
        denominator > 0,
        (
            df["market_return_sum"]
            - df["return_1h"]
        )
        / denominator,
        np.nan,
    )

    df = df.drop(
        columns=[
            "market_return_sum",
            "market_return_count",
        ]
    )

    return df


# ============================================================
# COMPLETE FEATURE BUILDER
# ============================================================

def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete feature-engineered DataFrame used for
    training or prediction.
    """

    df = validate_market_data(df)

    df = create_return_features(df)

    df = create_volatility_features(df)

    df = create_volume_features(df)

    df = create_market_context_feature(df)

    df = df.sort_values(
        ["timestamp", "coin_id"]
    ).reset_index(drop=True)

    return df


# ============================================================
# LATEST COMMON MARKET TIMESTAMP
# ============================================================

def get_latest_complete_timestamp(
    df: pd.DataFrame,
) -> pd.Timestamp:
    """
    Return the newest timestamp where:

    1. All configured cryptocurrencies are present.
    2. All model features are available for every coin.

    This prevents BTC being predicted using a timestamp where
    another coin required for market context is missing.
    """

    expected_coins = set(COINS)

    candidate_df = df.dropna(
        subset=ML_FEATURES
    ).copy()

    if candidate_df.empty:
        raise ValueError(
            "No rows contain a complete feature set."
        )

    timestamps = sorted(
        candidate_df["timestamp"].unique(),
        reverse=True,
    )

    for timestamp in timestamps:

        timestamp_rows = candidate_df[
            candidate_df["timestamp"]
            == timestamp
        ]

        available_coins = set(
            timestamp_rows["coin_id"]
            .astype(str)
            .tolist()
        )

        if expected_coins.issubset(
            available_coins
        ):
            return pd.Timestamp(timestamp)

    raise ValueError(
        "Could not find a timestamp containing complete "
        "features for all configured cryptocurrencies."
    )


# ============================================================
# GET LATEST FEATURE ROW
# ============================================================

def get_latest_feature_row(
    df: pd.DataFrame,
    coin_id: str,
) -> pd.DataFrame:
    """
    Return one production-ready feature row for a selected coin.
    """

    latest_timestamp = (
        get_latest_complete_timestamp(df)
    )

    row = df[
        (df["coin_id"] == coin_id)
        &
        (df["timestamp"] == latest_timestamp)
    ].copy()

    if row.empty:
        raise ValueError(
            f"No feature row found for {coin_id} "
            f"at {latest_timestamp}."
        )

    missing_features = (
        row[ML_FEATURES]
        .isna()
        .any()
    )

    missing_features = (
        missing_features[
            missing_features
        ]
        .index
        .tolist()
    )

    if missing_features:
        raise ValueError(
            f"Incomplete features for {coin_id}: "
            f"{missing_features}"
        )

    return row
