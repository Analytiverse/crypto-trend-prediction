import pandas as pd


INPUT_FILE = "data/raw/market_history.csv"
OUTPUT_FILE = "data/processed/market_hourly.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    print(f"Raw rows: {len(df)}")

    # Convert timestamp into proper UTC datetime
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    # Keep only columns required for the ML time-series dataset
    df = df[
        [
            "coin_id",
            "timestamp",
            "price",
            "market_cap",
            "total_volume",
        ]
    ].copy()

    # Normalize timestamps to the beginning of each hour
    df["timestamp"] = df["timestamp"].dt.floor("h")

    # Sort first so duplicate-hour handling is deterministic
    df = df.sort_values(
        ["coin_id", "timestamp"]
    )

    # If multiple observations exist for the same coin/hour,
    # keep the latest available observation
    df = df.drop_duplicates(
        subset=["coin_id", "timestamp"],
        keep="last"
    )

    # Remove rows with missing critical fields
    df = df.dropna(
        subset=[
            "coin_id",
            "timestamp",
            "price",
            "market_cap",
            "total_volume",
        ]
    )

    # Remove clearly invalid numerical values
    df = df[
        (df["price"] > 0)
        & (df["market_cap"] >= 0)
        & (df["total_volume"] >= 0)
    ]

    # Final chronological sorting
    df = df.sort_values(
        ["coin_id", "timestamp"]
    ).reset_index(drop=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Clean rows: {len(df)}")
    print(f"Removed rows: {len(pd.read_csv(INPUT_FILE)) - len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()