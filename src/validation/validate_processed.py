import pandas as pd


FILE_PATH = "data/processed/market_hourly.csv"


def main():
    df = pd.read_csv(FILE_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    print("\n========== DATASET OVERVIEW ==========")

    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\n========== ROWS PER COIN ==========")

    print(
        df.groupby("coin_id")
        .size()
        .sort_index()
    )

    print("\n========== MISSING VALUES ==========")

    print(df.isnull().sum())

    print("\n========== DUPLICATE COIN/HOUR ==========")

    duplicate_count = df.duplicated(
        subset=["coin_id", "timestamp"]
    ).sum()

    print(
        f"Duplicate coin/hour rows: {duplicate_count}"
    )

    print("\n========== INVALID VALUES ==========")

    print(
        "Price <= 0:",
        (df["price"] <= 0).sum()
    )

    print(
        "Market cap < 0:",
        (df["market_cap"] < 0).sum()
    )

    print(
        "Volume < 0:",
        (df["total_volume"] < 0).sum()
    )

    print("\n========== TIMESTAMP ALIGNMENT ==========")

    non_hour_aligned = df[
        (df["timestamp"].dt.minute != 0)
        | (df["timestamp"].dt.second != 0)
    ]

    print(
        f"Non-hour-aligned rows: "
        f"{len(non_hour_aligned)}"
    )

    print("\n========== HOURLY INTERVAL CHECK ==========")

    sorted_df = df.sort_values(
        ["coin_id", "timestamp"]
    ).copy()

    sorted_df["time_difference"] = (
        sorted_df
        .groupby("coin_id")["timestamp"]
        .diff()
    )

    non_hourly = sorted_df[
        sorted_df["time_difference"].notna()
        & (
            sorted_df["time_difference"]
            != pd.Timedelta(hours=1)
        )
    ]

    print(
        f"Non-hourly intervals: "
        f"{len(non_hourly)}"
    )

    if not non_hourly.empty:
        print(
            non_hourly[
                [
                    "coin_id",
                    "timestamp",
                    "time_difference"
                ]
            ].head(20)
        )


if __name__ == "__main__":
    main()