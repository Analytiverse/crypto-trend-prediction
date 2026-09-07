import pandas as pd

from config import PROCESSED_HISTORY_FILE


HORIZONS = [6, 12, 24]


def load_data():
    df = pd.read_csv(PROCESSED_HISTORY_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df.sort_values(
            ["coin_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    return df


def add_future_returns(df):
    result = df.copy()

    for horizon in HORIZONS:
        future_price = (
            result.groupby("coin_id")["price"]
            .shift(-horizon)
        )

        result[f"return_{horizon}h"] = (
            future_price / result["price"] - 1
        )

    return result


def print_dataset_overview(df):
    print(
        "\n========== DATASET OVERVIEW =========="
    )

    print(f"Total rows: {len(df)}")
    print(f"Coins: {df['coin_id'].nunique()}")

    print("\nRows per coin:")

    print(
        df.groupby("coin_id")
        .size()
    )

    print("\nDate range:")

    print(
        df.groupby("coin_id")["timestamp"]
        .agg(["min", "max"])
    )


def print_overall_return_distribution(df):
    print(
        "\n========== OVERALL RETURN DISTRIBUTION =========="
    )

    for horizon in HORIZONS:
        column = f"return_{horizon}h"

        returns = df[column].dropna()

        print(
            f"\n----- {horizon} HOUR RETURN -----"
        )

        print(
            returns.describe(
                percentiles=[
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


def print_return_distribution_by_coin(df):
    print(
        "\n========== RETURN DISTRIBUTION BY COIN =========="
    )

    for horizon in HORIZONS:
        column = f"return_{horizon}h"

        summary = (
            df.groupby("coin_id")[column]
            .agg(
                count="count",
                mean="mean",
                std="std",
                min="min",
                median="median",
                max="max",
            )
        )

        print(
            f"\n----- {horizon} HOUR BY COIN -----"
        )

        print(summary)


def print_absolute_movement(df):
    print(
        "\n========== ABSOLUTE PRICE MOVEMENT =========="
    )

    for horizon in HORIZONS:
        column = f"return_{horizon}h"

        absolute_returns = (
            df[column]
            .dropna()
            .abs()
        )

        print(
            f"\n----- {horizon} HOUR ABSOLUTE MOVEMENT -----"
        )

        print(
            absolute_returns.describe(
                percentiles=[
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                ]
            )
        )


def print_absolute_movement_by_coin(df):
    print(
        "\n========== ABSOLUTE MOVEMENT BY COIN =========="
    )

    for horizon in HORIZONS:
        column = f"return_{horizon}h"

        temp_df = df[
            ["coin_id", column]
        ].copy()

        temp_df["absolute_return"] = (
            temp_df[column].abs()
        )

        summary = (
            temp_df.groupby("coin_id")[
                "absolute_return"
            ]
            .agg(
                count="count",
                mean="mean",
                median="median",
                std="std",
                max="max",
            )
        )

        print(
            f"\n----- {horizon} HOUR BY COIN -----"
        )

        print(summary)


def main():
    df = load_data()

    print_dataset_overview(df)

    df = add_future_returns(df)

    print_overall_return_distribution(df)

    print_return_distribution_by_coin(df)

    print_absolute_movement(df)

    print_absolute_movement_by_coin(df)


if __name__ == "__main__":
    main()