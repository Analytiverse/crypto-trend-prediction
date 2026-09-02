import pandas as pd


FILE_PATH = "data/raw/market_history.csv"


def main():
    df = pd.read_csv(FILE_PATH)

    # Convert timestamp text into a proper datetime datatype
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

    print("\n========== DATE RANGE PER COIN ==========")

    date_ranges = df.groupby("coin_id")["timestamp"].agg(
        ["min", "max"]
    )

    print(date_ranges)

    print("\n========== MISSING VALUES ==========")

    print(df.isnull().sum())

    print("\n========== DUPLICATE CHECK ==========")

    duplicate_count = df.duplicated(
        subset=["coin_id", "timestamp_ms"]
    ).sum()

    print(
        f"Duplicate coin/timestamp records: "
        f"{duplicate_count}"
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

    print("\n========== TIMESTAMP INTERVALS ==========")

    sorted_df = df.sort_values(
        ["coin_id", "timestamp"]
    ).copy()

    sorted_df["time_difference"] = (
        sorted_df
        .groupby("coin_id")["timestamp"]
        .diff()
    )

    print(
        sorted_df["time_difference"]
        .value_counts()
        .head(10)
    )

    print("\n========== NON-HOURLY INTERVALS ==========")

    non_hourly = sorted_df[
        sorted_df["time_difference"].notna()
        & (
            sorted_df["time_difference"]
            != pd.Timedelta(hours=1)
        )
    ]

    print(
        f"Non-hourly intervals found: "
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