import pandas as pd


INPUT_FILE = "data/raw/market_history.csv"
OUTPUT_FILE = "data/processed/market_hourly.csv"


def clean_history_dataframe(df):
    """
    Clean and normalize historical market data.

    Accepts a DataFrame and returns a cleaned hourly DataFrame.
    """
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df[
        [
            "coin_id",
            "timestamp",
            "price",
            "market_cap",
            "total_volume",
        ]
    ].copy()

    # Normalize observations to the beginning of the hour
    df["timestamp"] = df["timestamp"].dt.floor("h")

    # Sort before duplicate handling
    df = df.sort_values(
        ["coin_id", "timestamp"]
    )

    # One observation per coin/hour
    df = df.drop_duplicates(
        subset=["coin_id", "timestamp"],
        keep="last",
    )

    # Remove missing critical values
    df = df.dropna(
        subset=[
            "coin_id",
            "timestamp",
            "price",
            "market_cap",
            "total_volume",
        ]
    )

    # Remove invalid numerical values
    df = df[
        (df["price"] > 0)
        & (df["market_cap"] >= 0)
        & (df["total_volume"] >= 0)
    ]

    return df.sort_values(
        ["coin_id", "timestamp"]
    ).reset_index(drop=True)


def main():
    raw_df = pd.read_csv(INPUT_FILE)

    print(f"Raw rows: {len(raw_df)}")

    clean_df = clean_history_dataframe(raw_df)

    clean_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Clean rows: {len(clean_df)}")
    print(f"Removed rows: {len(raw_df) - len(clean_df)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()