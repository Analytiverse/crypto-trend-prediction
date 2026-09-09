"""
Exploratory Data Analysis for cryptocurrency market data.

This module loads cleaned hourly market data from PostgreSQL
and performs exploratory analysis before feature engineering
and model development.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import text

from src.database.connection import engine


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RETURN_HORIZONS = [1, 6, 12, 24]

OUTPUT_DIR = Path("outputs/eda")


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_market_data() -> pd.DataFrame:
    """
    Load cleaned hourly market data from PostgreSQL.
    """

    query = text("""
        SELECT
            c.coin_id,
            c.symbol,
            mh.timestamp,
            mh.price,
            mh.market_cap,
            mh.total_volume
        FROM market_hourly AS mh
        JOIN coins AS c
            ON mh.coin_id = c.coin_id
        ORDER BY c.coin_id, mh.timestamp
    """)

    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    return df


# ---------------------------------------------------------
# Basic inspection
# ---------------------------------------------------------

def inspect_data(df: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.
    """

    print("\nDataset shape")
    print("-" * 40)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns")
    print("-" * 40)
    print(df.columns.tolist())

    print("\nData types")
    print("-" * 40)
    print(df.dtypes)

    print("\nFirst 5 rows")
    print("-" * 40)
    print(df.head())

    print("\nDate range")
    print("-" * 40)
    print(f"Start: {df['timestamp'].min()}")
    print(f"End:   {df['timestamp'].max()}")

    print("\nRows per coin")
    print("-" * 40)
    print(df.groupby("coin_id").size())

    print("\nMissing values")
    print("-" * 40)
    print(df.isna().sum())


# ---------------------------------------------------------
# Data quality
# ---------------------------------------------------------

def check_data_quality(df: pd.DataFrame) -> None:
    """
    Check duplicates, invalid values, and time gaps.
    """

    print("\n" + "=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)

    duplicate_count = df.duplicated(
        subset=["coin_id", "timestamp"]
    ).sum()

    print("\nDuplicate coin/timestamp rows")
    print("-" * 40)
    print(duplicate_count)

    print("\nInvalid numeric values")
    print("-" * 40)

    print(f"Price <= 0:        {(df['price'] <= 0).sum()}")
    print(f"Market cap <= 0:   {(df['market_cap'] <= 0).sum()}")
    print(f"Volume < 0:        {(df['total_volume'] < 0).sum()}")

    ordered_df = df.sort_values(
        ["coin_id", "timestamp"]
    ).copy()

    ordered_df["time_diff"] = (
        ordered_df
        .groupby("coin_id")["timestamp"]
        .diff()
    )

    print("\nTime interval distribution")
    print("-" * 40)

    print(
        ordered_df["time_diff"]
        .value_counts()
        .sort_index()
    )

    gaps = ordered_df[
        ordered_df["time_diff"] > pd.Timedelta(hours=1)
    ]

    print("\nGaps larger than 1 hour")
    print("-" * 40)
    print(f"Number of gaps: {len(gaps)}")

    if not gaps.empty:
        print(
            gaps[
                ["coin_id", "timestamp", "time_diff"]
            ].to_string(index=False)
        )


# ---------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------

def descriptive_statistics(df: pd.DataFrame) -> None:
    """
    Display descriptive statistics for raw market variables.
    """

    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    numeric_columns = [
        "price",
        "market_cap",
        "total_volume",
    ]

    stats = (
        df
        .groupby("coin_id")[numeric_columns]
        .describe()
    )

    print(stats)


# ---------------------------------------------------------
# Return calculation
# ---------------------------------------------------------

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate historical returns using exact timestamps.

    Exact timestamp matching prevents gaps in the hourly
    time series from creating incorrect return calculations.
    """

    print("\n" + "=" * 60)
    print("CALCULATING RETURNS")
    print("=" * 60)

    df = df.copy()

    for hours in RETURN_HORIZONS:

        previous_prices = df[
            ["coin_id", "timestamp", "price"]
        ].copy()

        previous_prices["timestamp"] = (
            previous_prices["timestamp"]
            + pd.Timedelta(hours=hours)
        )

        previous_prices = previous_prices.rename(
            columns={
                "price": f"price_{hours}h_ago"
            }
        )

        df = df.merge(
            previous_prices,
            on=["coin_id", "timestamp"],
            how="left"
        )

        df[f"return_{hours}h"] = (
            df["price"] /
            df[f"price_{hours}h_ago"]
            - 1
        )

    return_columns = [
        f"return_{hours}h"
        for hours in RETURN_HORIZONS
    ]

    print("\nReturn columns created")
    print("-" * 40)
    print(return_columns)

    print("\nMissing returns")
    print("-" * 40)
    print(df[return_columns].isna().sum())

    return df


# ---------------------------------------------------------
# Basic return analysis
# ---------------------------------------------------------

def analyze_returns(df: pd.DataFrame) -> None:
    """
    Display descriptive statistics for returns.
    """

    print("\n" + "=" * 60)
    print("RETURN ANALYSIS")
    print("=" * 60)

    for hours in RETURN_HORIZONS:

        column = f"return_{hours}h"

        print(f"\n{column.upper()}")
        print("-" * 40)

        stats = (
            df
            .groupby("coin_id")[column]
            .agg([
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ])
        )

        print(stats)


# ---------------------------------------------------------
# Return distribution analysis
# ---------------------------------------------------------

def analyze_return_distributions(df: pd.DataFrame) -> None:
    """
    Analyze percentiles, skewness, and kurtosis of returns.
    """

    print("\n" + "=" * 60)
    print("RETURN DISTRIBUTION ANALYSIS")
    print("=" * 60)

    percentiles = [
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
    ]

    for hours in RETURN_HORIZONS:

        column = f"return_{hours}h"

        print(f"\n{column.upper()}")
        print("-" * 60)

        # Percentiles
        percentile_stats = (
            df
            .groupby("coin_id")[column]
            .quantile(percentiles)
            .unstack()
        )

        percentile_stats.columns = [
            "1%",
            "5%",
            "25%",
            "50%",
            "75%",
            "95%",
            "99%",
        ]

        print("\nPercentiles")
        print(percentile_stats)

        # Skewness
        skewness = (
            df
            .groupby("coin_id")[column]
            .skew()
        )

        print("\nSkewness")
        print(skewness)

        # Kurtosis
        kurtosis = (
            df
            .groupby("coin_id")[column]
            .apply(lambda x: x.kurt())
        )

        print("\nKurtosis")
        print(kurtosis)


# ---------------------------------------------------------
# Histograms
# ---------------------------------------------------------

def plot_return_distributions(df: pd.DataFrame) -> None:
    """
    Create one return-distribution histogram per horizon.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("CREATING RETURN HISTOGRAMS")
    print("=" * 60)

    for hours in RETURN_HORIZONS:

        column = f"return_{hours}h"

        plt.figure(figsize=(10, 6))

        for coin_id, coin_data in df.groupby("coin_id"):

            returns = (
                coin_data[column]
                .dropna()
                * 100
            )

            plt.hist(
                returns,
                bins=80,
                alpha=0.35,
                label=coin_id,
            )

        plt.title(
            f"{hours}-Hour Return Distribution"
        )

        plt.xlabel("Return (%)")
        plt.ylabel("Frequency")

        plt.legend()

        plt.tight_layout()

        output_file = (
            OUTPUT_DIR /
            f"return_{hours}h_distribution.png"
        )

        plt.savefig(
            output_file,
            dpi=150
        )

        plt.close()

        print(f"Saved: {output_file}")

def analyze_rolling_volatility(df):
    """
    Analyze how cryptocurrency volatility changes over time.

    Rolling volatility is calculated using 1-hour returns.
    """

    print("\n" + "=" * 60)
    print("ROLLING VOLATILITY ANALYSIS")
    print("=" * 60)

    df = df.copy()

    # Sort chronologically before applying rolling windows.
    df = df.sort_values(
        ["coin_id", "timestamp"]
    ).copy()

    # 24-hour rolling volatility
    df["volatility_24h"] = (
        df.groupby("coin_id")["return_1h"]
        .transform(
            lambda x: x.rolling(
                window=24,
                min_periods=24,
            ).std()
        )
    )

    # 72-hour rolling volatility
    df["volatility_72h"] = (
        df.groupby("coin_id")["return_1h"]
        .transform(
            lambda x: x.rolling(
                window=72,
                min_periods=72,
            ).std()
        )
    )

    print("\n24H ROLLING VOLATILITY")
    print("-" * 60)

    summary_24h = (
        df.groupby("coin_id")["volatility_24h"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(summary_24h)

    print("\n72H ROLLING VOLATILITY")
    print("-" * 60)

    summary_72h = (
        df.groupby("coin_id")["volatility_72h"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(summary_72h)

    return df

def plot_rolling_volatility(df):
    """
    Plot 24-hour rolling volatility for each cryptocurrency.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("CREATING ROLLING VOLATILITY PLOTS")
    print("=" * 60)

    for coin_id, coin_data in df.groupby("coin_id"):

        plt.figure(figsize=(12, 6))

        plt.plot(
            coin_data["timestamp"],
            coin_data["volatility_24h"] * 100,
            label="24h volatility",
        )

        plt.plot(
            coin_data["timestamp"],
            coin_data["volatility_72h"] * 100,
            label="72h volatility",
        )

        plt.title(
            f"{coin_id.capitalize()} Rolling Volatility"
        )

        plt.xlabel("Time")
        plt.ylabel("Volatility (%)")

        plt.legend()
        plt.tight_layout()

        output_file = (
            OUTPUT_DIR /
            f"{coin_id}_rolling_volatility.png"
        )

        plt.savefig(
            output_file,
            dpi=150
        )

        plt.close()

        print(f"Saved: {output_file}")

def analyze_return_autocorrelation(df):
    """
    Measure autocorrelation of 1-hour returns at different lags.

    Positive correlation may indicate short-term momentum.
    Negative correlation may indicate short-term mean reversion.
    Values close to zero indicate little linear relationship.
    """

    print("\n" + "=" * 60)
    print("RETURN AUTOCORRELATION ANALYSIS")
    print("=" * 60)

    lags = [1, 2, 3, 6, 12, 24]

    results = []

    for coin_id, coin_data in df.groupby("coin_id"):

        coin_data = coin_data.sort_values(
            "timestamp"
        )

        returns = (
            coin_data["return_1h"]
            .dropna()
        )

        row = {
            "coin_id": coin_id
        }

        for lag in lags:

            row[f"lag_{lag}h"] = (
                returns.autocorr(lag=lag)
            )

        results.append(row)

    autocorrelation_df = pd.DataFrame(
        results
    ).set_index("coin_id")

    print("\n1-HOUR RETURN AUTOCORRELATION")
    print("-" * 60)

    print(autocorrelation_df)

    return autocorrelation_df

def analyze_coin_correlations(df):
    """
    Analyze correlation between hourly returns
    of different cryptocurrencies.
    """

    print("\n" + "=" * 60)
    print("CROSS-COIN RETURN CORRELATION")
    print("=" * 60)

    return_matrix = (
        df.pivot(
            index="timestamp",
            columns="coin_id",
            values="return_1h",
        )
    )

    correlation_matrix = (
        return_matrix.corr()
    )

    print("\n1-HOUR RETURN CORRELATION MATRIX")
    print("-" * 60)

    print(correlation_matrix)

    return correlation_matrix
def calculate_future_returns(df):
    """
    Calculate future returns for the prediction horizons.

    These will later be used to create the
    UP / STABLE / DOWN target labels.
    """

    print("\n" + "=" * 60)
    print("CALCULATING FUTURE RETURNS")
    print("=" * 60)

    df = df.copy()

    for hours in [6, 12, 24]:

        future_prices = df[
            ["coin_id", "timestamp", "price"]
        ].copy()

        # Move future prices backward so they align
        # with the current timestamp.
        future_prices["timestamp"] = (
            future_prices["timestamp"]
            - pd.Timedelta(hours=hours)
        )

        future_prices = future_prices.rename(
            columns={
                "price": f"price_future_{hours}h"
            }
        )

        df = df.merge(
            future_prices,
            on=["coin_id", "timestamp"],
            how="left",
        )

        df[f"future_return_{hours}h"] = (
            df[f"price_future_{hours}h"]
            / df["price"]
            - 1
        )

    future_columns = [
        "future_return_6h",
        "future_return_12h",
        "future_return_24h",
    ]

    print("\nFuture return columns created")
    print("-" * 60)
    print(future_columns)

    print("\nMissing future returns")
    print("-" * 60)
    print(df[future_columns].isna().sum())

    return df

def analyze_target_thresholds(df):
    """
    Test candidate thresholds for creating
    DOWN / STABLE / UP target classes.
    """

    print("\n" + "=" * 60)
    print("TARGET THRESHOLD ANALYSIS")
    print("=" * 60)

    thresholds = [
        0.005,   # 0.5%
        0.010,   # 1.0%
        0.015,   # 1.5%
        0.020,   # 2.0%
        0.025,   # 2.5%
        0.030,   # 3.0%
    ]

    for hours in [6, 12, 24]:

        column = f"future_return_{hours}h"

        valid_returns = df[column].dropna()

        print("\n" + "=" * 60)
        print(f"{hours}-HOUR TARGET")
        print("=" * 60)

        print("\nFuture return percentiles")
        print("-" * 60)

        print(
            valid_returns.quantile(
                [
                    0.05,
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                ]
            )
        )

        print("\nCandidate threshold class distributions")
        print("-" * 60)

        for threshold in thresholds:

            down = (
                valid_returns < -threshold
            ).mean() * 100

            stable = (
                (
                    valid_returns >= -threshold
                )
                &
                (
                    valid_returns <= threshold
                )
            ).mean() * 100

            up = (
                valid_returns > threshold
            ).mean() * 100

            print(
                f"\nThreshold: ±{threshold * 100:.1f}%"
            )

            print(
                f"DOWN:   {down:6.2f}%"
            )

            print(
                f"STABLE: {stable:6.2f}%"
            )

            print(
                f"UP:     {up:6.2f}%"
            )
def analyze_thresholds_by_coin(df):
    """
    Check the proposed target thresholds separately
    for each cryptocurrency.
    """

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION BY COIN")
    print("=" * 60)

    proposed_thresholds = {
        6: 0.005,
        12: 0.010,
        24: 0.015,
    }

    for hours, threshold in proposed_thresholds.items():

        column = f"future_return_{hours}h"

        print("\n" + "=" * 60)
        print(
            f"{hours}H TARGET - THRESHOLD ±{threshold * 100:.1f}%"
        )
        print("=" * 60)

        results = []

        for coin_id, coin_data in df.groupby("coin_id"):

            returns = coin_data[column].dropna()

            down = (returns < -threshold).mean() * 100

            stable = (
                (returns >= -threshold)
                & (returns <= threshold)
            ).mean() * 100

            up = (returns > threshold).mean() * 100

            results.append(
                {
                    "coin_id": coin_id,
                    "DOWN": down,
                    "STABLE": stable,
                    "UP": up,
                }
            )

        result_df = pd.DataFrame(results).set_index("coin_id")

        print(result_df.round(2))

def create_target_labels(df):
    """
    Create UP / DOWN / STABLE target labels
    using the selected horizon-specific thresholds.
    """

    print("\n" + "=" * 60)
    print("CREATING TARGET LABELS")
    print("=" * 60)

    df = df.copy()

    thresholds = {
        6: 0.005,
        12: 0.010,
        24: 0.015,
    }

    for hours, threshold in thresholds.items():

        return_column = f"future_return_{hours}h"
        target_column = f"target_{hours}h"

        # Start with missing targets.
        # Rows without a known future price must NOT
        # accidentally become STABLE.
        df[target_column] = pd.NA

        valid = df[return_column].notna()

        df.loc[
            valid & (df[return_column] > threshold),
            target_column,
        ] = "UP"

        df.loc[
            valid & (df[return_column] < -threshold),
            target_column,
        ] = "DOWN"

        df.loc[
            valid
            & (df[return_column] >= -threshold)
            & (df[return_column] <= threshold),
            target_column,
        ] = "STABLE"

    target_columns = [
        "target_6h",
        "target_12h",
        "target_24h",
    ]

    print("\nTarget columns created")
    print("-" * 60)
    print(target_columns)

    for target in target_columns:

        print(f"\n{target.upper()}")
        print("-" * 60)

        print(df[target].value_counts(dropna=False))

        print("\nPercentages")
        print(
            (
                df[target]
                .value_counts(normalize=True, dropna=True)
                * 100
            ).round(2)
        )

    return df
def create_volume_features(df):
    """
    Create volume-change features using only
    information available at or before each timestamp.
    """

    print("\n" + "=" * 60)
    print("CREATING VOLUME FEATURES")
    print("=" * 60)

    df = df.copy()

    for hours in [1, 6, 12, 24]:

        previous_volume = df[
            ["coin_id", "timestamp", "total_volume"]
        ].copy()

        previous_volume["timestamp"] = (
            previous_volume["timestamp"]
            + pd.Timedelta(hours=hours)
        )

        previous_volume = previous_volume.rename(
            columns={
                "total_volume":
                f"volume_previous_{hours}h"
            }
        )

        df = df.merge(
            previous_volume,
            on=["coin_id", "timestamp"],
            how="left",
        )

        df[f"volume_change_{hours}h"] = (
            df["total_volume"]
            / df[f"volume_previous_{hours}h"]
            - 1
        )

    feature_columns = [
        "volume_change_1h",
        "volume_change_6h",
        "volume_change_12h",
        "volume_change_24h",
    ]

    print("\nVolume features created")
    print("-" * 60)
    print(feature_columns)

    print("\nMissing values")
    print("-" * 60)
    print(df[feature_columns].isna().sum())

    print("\nVolume feature statistics")
    print("-" * 60)

    print(
        df.groupby("coin_id")[feature_columns]
        .agg(["mean", "median", "std", "min", "max"])
    )

    return df

def create_log_volume_features(df):
    """
    Create log-volume-change features.

    Log ratios reduce the impact of extreme percentage
    changes while preserving the direction and magnitude
    of changes in trading activity.
    """

    print("\n" + "=" * 60)
    print("CREATING LOG VOLUME FEATURES")
    print("=" * 60)

    df = df.copy()

    for hours in [1, 6, 12, 24]:

        previous_column = f"volume_previous_{hours}h"
        feature_column = f"log_volume_change_{hours}h"

        valid = (
            (df["total_volume"] > 0)
            & (df[previous_column] > 0)
        )

        df[feature_column] = np.nan

        df.loc[valid, feature_column] = np.log(
            df.loc[valid, "total_volume"]
            / df.loc[valid, previous_column]
        )

    feature_columns = [
        "log_volume_change_1h",
        "log_volume_change_6h",
        "log_volume_change_12h",
        "log_volume_change_24h",
    ]

    print("\nLog volume features created")
    print("-" * 60)
    print(feature_columns)

    print("\nMissing values")
    print("-" * 60)
    print(df[feature_columns].isna().sum())

    print("\nLog volume feature statistics")
    print("-" * 60)

    print(
        df.groupby("coin_id")[feature_columns]
        .agg(["mean", "median", "std", "min", "max"])
    )

    return df
def create_market_context_features(df):
    """
    Create market-wide context using the returns of the
    other cryptocurrencies at the same timestamp.

    The current coin is excluded from its own market
    context feature.
    """

    print("\n" + "=" * 60)
    print("CREATING MARKET CONTEXT FEATURES")
    print("=" * 60)

    df = df.copy()

    market_stats = (
        df.groupby("timestamp")["return_1h"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": "market_return_sum",
                "count": "market_return_count",
            }
        )
    )

    df = df.merge(
        market_stats,
        on="timestamp",
        how="left",
    )

    df["other_coins_return_1h"] = (
        (df["market_return_sum"] - df["return_1h"])
        / (df["market_return_count"] - 1)
    )

    # Temporary calculation columns are no longer needed.
    df = df.drop(
        columns=[
            "market_return_sum",
            "market_return_count",
        ]
    )

    print("\nMarket context features created")
    print("-" * 60)
    print(["other_coins_return_1h"])

    print("\nMissing values")
    print("-" * 60)
    print(df[["other_coins_return_1h"]].isna().sum())

    print("\nOther-coins return statistics")
    print("-" * 60)
    print(df["other_coins_return_1h"].describe())

    return df
def create_market_cap_features(df):
    """
    Create log market-cap change features using exact
    historical timestamp alignment.
    """

    print("\n" + "=" * 60)
    print("CREATING MARKET CAP FEATURES")
    print("=" * 60)

    df = df.copy()

    horizons = [1, 6, 12, 24]
    created_columns = []

    for hours in horizons:
        previous = df[
            ["coin_id", "timestamp", "market_cap"]
        ].copy()

        previous["timestamp"] = (
            previous["timestamp"]
            + pd.Timedelta(hours=hours)
        )

        previous = previous.rename(
            columns={
                "market_cap": f"market_cap_previous_{hours}h"
            }
        )

        df = df.merge(
            previous,
            on=["coin_id", "timestamp"],
            how="left",
        )

        previous_col = f"market_cap_previous_{hours}h"
        feature_col = f"log_market_cap_change_{hours}h"

        valid = (
            (df["market_cap"] > 0)
            & (df[previous_col] > 0)
        )

        df[feature_col] = np.nan

        df.loc[valid, feature_col] = np.log(
            df.loc[valid, "market_cap"]
            / df.loc[valid, previous_col]
        )

        created_columns.append(feature_col)

        # Previous market cap was only needed for calculation.
        df = df.drop(columns=[previous_col])

    print("\nMarket cap features created")
    print("-" * 60)
    print(created_columns)

    print("\nMissing values")
    print("-" * 60)
    print(df[created_columns].isna().sum())

    print("\nMarket cap feature statistics")
    print("-" * 60)

    stats = (
        df.groupby("coin_id")[created_columns]
        .agg(["mean", "median", "std", "min", "max"])
    )

    print(stats)

    return df

def analyze_market_cap_feature_redundancy(df):
    """
    Check whether market-cap change features provide
    information different from price-return features.
    """

    print("\n" + "=" * 60)
    print("MARKET CAP FEATURE REDUNDANCY CHECK")
    print("=" * 60)

    horizons = [1, 6, 12, 24]

    for hours in horizons:
        return_col = f"return_{hours}h"
        market_cap_col = f"log_market_cap_change_{hours}h"

        correlation = df[
            [return_col, market_cap_col]
        ].corr().iloc[0, 1]

        print(
            f"{hours}h price return vs market-cap change: "
            f"{correlation:.6f}"
        )
def validate_final_features(df):
    """
    Validate the candidate ML feature set before
    constructing the modeling dataset.
    """

    print("\n" + "=" * 60)
    print("FINAL FEATURE VALIDATION")
    print("=" * 60)

    feature_columns = [
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

    print("\nFinal candidate features")
    print("-" * 60)

    for feature in feature_columns:
        print(feature)

    missing_columns = [
        col for col in feature_columns
        if col not in df.columns
    ]

    print("\nMissing feature columns")
    print("-" * 60)
    print(missing_columns)

    if missing_columns:
        return

    print("\nMissing values by feature")
    print("-" * 60)
    print(df[feature_columns].isna().sum())

    print("\nInfinite values by feature")
    print("-" * 60)

    infinite_counts = pd.Series({
        col: np.isinf(df[col]).sum()
        for col in feature_columns
    })

    print(infinite_counts)

    valid_feature_rows = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    print("\nUsable rows after feature warm-up")
    print("-" * 60)
    print(f"Total rows:  {len(df)}")
    print(f"Usable rows: {len(valid_feature_rows)}")
    print(f"Removed:     {len(df) - len(valid_feature_rows)}")

    print("\nFeature correlation matrix")
    print("-" * 60)
    print(df[feature_columns].corr())

    return feature_columns
def validate_feature_leakage(feature_columns):
    """
    Ensure future-looking target information is not
    accidentally included in the ML feature set.
    """

    print("\n" + "=" * 60)
    print("FEATURE LEAKAGE CHECK")
    print("=" * 60)

    forbidden_columns = [
        "future_return_6h",
        "future_return_12h",
        "future_return_24h",
        "target_6h",
        "target_12h",
        "target_24h",
    ]

    leaked_columns = [
        col for col in feature_columns
        if col in forbidden_columns
    ]

    print("\nForbidden columns found in features")
    print("-" * 60)
    print(leaked_columns)

    if leaked_columns:
        print("\n❌ LEAKAGE DETECTED")
    else:
        print("\n✅ No target leakage detected")
# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    """
    Run the complete exploratory data analysis pipeline.
    """

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    print("Loading market data from PostgreSQL...")

    df = load_market_data()

    print(f"Loaded {len(df):,} rows.")

    # -----------------------------------------------------
    # Stage 1 - Data understanding
    # -----------------------------------------------------

    inspect_data(df)

    # -----------------------------------------------------
    # Stage 2 - Data quality
    # -----------------------------------------------------

    check_data_quality(df)

    # -----------------------------------------------------
    # Descriptive statistics
    # -----------------------------------------------------

    descriptive_statistics(df)

    # -----------------------------------------------------
    # Stage 3 - Return calculation and analysis
    # -----------------------------------------------------

    df = calculate_returns(df)

    analyze_returns(df)

    # -----------------------------------------------------
    # Stage 4 - Return distribution analysis
    # -----------------------------------------------------

    analyze_return_distributions(df)

    plot_return_distributions(df)

    # -----------------------------------------------------
    # Stage 5 - Time-series analysis
    # -----------------------------------------------------

    df = analyze_rolling_volatility(df)
    plot_rolling_volatility(df)
    analyze_return_autocorrelation(df)
    # -----------------------------------------------------
    # Stage 6 - Relationship analysis
    # -----------------------------------------------------

    analyze_coin_correlations(df)

    # -----------------------------------------------------
    # Stage 7 - Target analysis
    # -----------------------------------------------------
    df = calculate_future_returns(df)

    analyze_target_thresholds(df)
    analyze_thresholds_by_coin(df)
    df = create_target_labels(df)
    # -----------------------------------------------------
    # Stage 8 - Feature engineering
    # -----------------------------------------------------
  

    df = create_volume_features(df)
    df = create_log_volume_features(df)
    df = create_market_context_features(df)
   
    df = create_market_cap_features(df)
    analyze_market_cap_feature_redundancy(df)
    feature_columns = validate_final_features(df)
    validate_feature_leakage(feature_columns)
if __name__ == "__main__":
    main()