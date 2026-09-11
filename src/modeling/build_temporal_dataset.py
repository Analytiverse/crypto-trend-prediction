"""
Temporal Modeling Dataset Builder
=================================

This module prepares the already feature-engineered cryptocurrency
dataset for machine-learning modeling.

Important:
- PostgreSQL remains the source of truth.
- No CSV file is read here.
- No CSV file is written here.
- The DataFrame is received from analyze_returns.py after targets,
  feature engineering, feature validation, and leakage validation.
- Data is split chronologically, never randomly.
- A horizon-specific purge is applied at split boundaries to prevent
  future target windows from leaking into the next split.

Example:
If target_24h at time t uses price at t + 24 hours,
then the final 24 hours of Train must be removed so that
training labels do not depend on Validation-period prices.
"""

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

TARGET_COLUMNS = [
    "target_6h",
    "target_12h",
    "target_24h",
]

TARGET_HORIZONS = {
    "target_6h": 6,
    "target_12h": 12,
    "target_24h": 24,
}


# ============================================================
# VALIDATE SPLIT RATIOS
# ============================================================

def validate_split_ratios():
    """
    Ensure Train + Validation + Test = 100%.
    """

    total = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if not np.isclose(total, 1.0):
        raise ValueError(
            "Train, validation and test ratios must add to 1.0. "
            f"Current total: {total}"
        )


# ============================================================
# PREPARE MODELING DATAFRAME
# ============================================================

def prepare_modeling_dataframe(
    df,
    feature_columns,
):
    """
    Prepare the already feature-engineered DataFrame
    for temporal modeling.
    """

    print("\n" + "=" * 70)
    print("TEMPORAL MODELING DATASET")
    print("=" * 70)

    if df is None:
        raise ValueError(
            "Input dataframe is None."
        )

    if df.empty:
        raise ValueError(
            "Input dataframe is empty."
        )

    if not feature_columns:
        raise ValueError(
            "No validated feature columns were supplied."
        )

    modeling_df = df.copy()

    print(
        f"\nRows received: "
        f"{len(modeling_df):,}"
    )

    print(
        f"Columns received: "
        f"{len(modeling_df.columns)}"
    )

    print(
        f"Validated ML features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Required identifier columns
    # --------------------------------------------------------

    required_columns = [
        "coin_id",
        "timestamp",
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in modeling_df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Missing required columns: "
            f"{missing_required}"
        )

    # --------------------------------------------------------
    # Required targets
    # --------------------------------------------------------

    missing_targets = [
        column
        for column in TARGET_COLUMNS
        if column not in modeling_df.columns
    ]

    if missing_targets:
        raise ValueError(
            f"Missing target columns: "
            f"{missing_targets}"
        )

    # --------------------------------------------------------
    # Required features
    # --------------------------------------------------------

    missing_features = [
        column
        for column in feature_columns
        if column not in modeling_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Timestamp conversion
    # --------------------------------------------------------

    modeling_df["timestamp"] = pd.to_datetime(
        modeling_df["timestamp"],
        utc=True,
        errors="coerce",
    )

    invalid_timestamps = (
        modeling_df["timestamp"]
        .isna()
        .sum()
    )

    print(
        f"Invalid timestamps: "
        f"{invalid_timestamps}"
    )

    if invalid_timestamps > 0:
        modeling_df = modeling_df.dropna(
            subset=["timestamp"]
        )

    # --------------------------------------------------------
    # Replace infinite feature values
    # --------------------------------------------------------

    modeling_df[feature_columns] = (
        modeling_df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Remove feature warm-up rows
    # --------------------------------------------------------

    rows_before = len(
        modeling_df
    )

    modeling_df = modeling_df.dropna(
        subset=feature_columns
    )

    rows_after = len(
        modeling_df
    )

    print(
        "Rows removed because features "
        f"are incomplete: "
        f"{rows_before - rows_after:,}"
    )

    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

    modeling_df = (
        modeling_df
        .sort_values(
            [
                "timestamp",
                "coin_id",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Usable modeling rows: "
        f"{len(modeling_df):,}"
    )

    print(
        "Date range: "
        f"{modeling_df['timestamp'].min()} "
        "-> "
        f"{modeling_df['timestamp'].max()}"
    )

    return modeling_df


# ============================================================
# CREATE INITIAL TEMPORAL SPLIT
# ============================================================

def create_initial_temporal_split(
    df,
):
    """
    Create chronological Train / Validation / Test splits
    using unique timestamps.

    Important:
    All coins at the same timestamp remain in the same split.
    """

    if df.empty:
        raise ValueError(
            "Cannot split an empty dataframe."
        )

    unique_timestamps = (
        df["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    timestamp_count = len(
        unique_timestamps
    )

    if timestamp_count < 3:
        raise ValueError(
            "Not enough unique timestamps "
            "for train/validation/test splits."
        )

    train_end_index = int(
        timestamp_count
        * TRAIN_RATIO
    )

    validation_end_index = int(
        timestamp_count
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    if train_end_index <= 0:
        raise ValueError(
            "Training split contains "
            "no timestamps."
        )

    if validation_end_index <= train_end_index:
        raise ValueError(
            "Validation split contains "
            "no timestamps."
        )

    if validation_end_index >= timestamp_count:
        raise ValueError(
            "Test split contains "
            "no timestamps."
        )

    train_timestamps = (
        unique_timestamps.iloc[
            :train_end_index
        ]
    )

    validation_timestamps = (
        unique_timestamps.iloc[
            train_end_index:
            validation_end_index
        ]
    )

    test_timestamps = (
        unique_timestamps.iloc[
            validation_end_index:
        ]
    )

    train_df = (
        df[
            df["timestamp"].isin(
                train_timestamps
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    validation_df = (
        df[
            df["timestamp"].isin(
                validation_timestamps
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        df[
            df["timestamp"].isin(
                test_timestamps
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# APPLY PURGE
# ============================================================

def apply_horizon_purge(
    train_df,
    validation_df,
    test_df,
    horizon_hours,
):
    """
    Remove rows near temporal boundaries whose future
    target window crosses into the next split.

    Example for a 24h target:

    If Validation begins at:
        2026-08-12 11:00

    then a training row at:
        2026-08-12 10:00

    would use the future price at:
        2026-08-13 10:00

    which lies inside Validation.

    Therefore the final 24 hours of Train are removed.

    The same logic is applied between Validation and Test.
    """

    print("\nApplying temporal purge")
    print("-" * 70)

    print(
        f"Prediction horizon: "
        f"{horizon_hours} hours"
    )

    if train_df.empty:
        raise ValueError(
            "Train dataframe is empty "
            "before purge."
        )

    if validation_df.empty:
        raise ValueError(
            "Validation dataframe is empty "
            "before purge."
        )

    if test_df.empty:
        raise ValueError(
            "Test dataframe is empty "
            "before purge."
        )

    purge_delta = pd.Timedelta(
        hours=horizon_hours
    )

    # --------------------------------------------------------
    # Boundary timestamps
    # --------------------------------------------------------

    validation_start = (
        validation_df["timestamp"]
        .min()
    )

    test_start = (
        test_df["timestamp"]
        .min()
    )

    # --------------------------------------------------------
    # Training cutoff
    #
    # Keep only rows satisfying:
    #
    #     timestamp + horizon < validation_start
    #
    # Equivalent:
    #
    #     timestamp < validation_start - horizon
    #
    # --------------------------------------------------------

    train_cutoff = (
        validation_start
        - purge_delta
    )

    # --------------------------------------------------------
    # Validation cutoff
    #
    # Keep only rows satisfying:
    #
    #     timestamp + horizon < test_start
    #
    # --------------------------------------------------------

    validation_cutoff = (
        test_start
        - purge_delta
    )

    train_rows_before = len(
        train_df
    )

    validation_rows_before = len(
        validation_df
    )

    test_rows_before = len(
        test_df
    )

    # Strictly less than cutoff.
    # This guarantees target_end < next_split_start.

    purged_train_df = (
        train_df[
            train_df["timestamp"]
            < train_cutoff
        ]
        .copy()
        .reset_index(drop=True)
    )

    purged_validation_df = (
        validation_df[
            validation_df["timestamp"]
            < validation_cutoff
        ]
        .copy()
        .reset_index(drop=True)
    )

    # Test does not need a boundary purge because
    # there is no later evaluation split.
    #
    # Rows whose future targets are unavailable were already
    # removed before splitting.

    purged_test_df = (
        test_df
        .copy()
        .reset_index(drop=True)
    )

    train_removed = (
        train_rows_before
        - len(purged_train_df)
    )

    validation_removed = (
        validation_rows_before
        - len(purged_validation_df)
    )

    print(
        f"\nValidation starts: "
        f"{validation_start}"
    )

    print(
        f"Test starts:       "
        f"{test_start}"
    )

    print(
        f"\nTrain purge cutoff: "
        f"{train_cutoff}"
    )

    print(
        f"Validation purge cutoff: "
        f"{validation_cutoff}"
    )

    print(
        f"\nTrain rows before purge: "
        f"{train_rows_before:,}"
    )

    print(
        f"Train rows removed:      "
        f"{train_removed:,}"
    )

    print(
        f"Train rows after purge:  "
        f"{len(purged_train_df):,}"
    )

    print(
        f"\nValidation rows before purge: "
        f"{validation_rows_before:,}"
    )

    print(
        f"Validation rows removed:      "
        f"{validation_removed:,}"
    )

    print(
        f"Validation rows after purge:  "
        f"{len(purged_validation_df):,}"
    )

    print(
        f"\nTest rows unchanged: "
        f"{test_rows_before:,}"
    )

    if purged_train_df.empty:
        raise ValueError(
            "Training dataframe became empty "
            "after applying purge."
        )

    if purged_validation_df.empty:
        raise ValueError(
            "Validation dataframe became empty "
            "after applying purge."
        )

    return (
        purged_train_df,
        purged_validation_df,
        purged_test_df,
    )


# ============================================================
# VALIDATE BASIC TEMPORAL ORDER
# ============================================================

def validate_temporal_order(
    train_df,
    validation_df,
    test_df,
):
    """
    Verify that raw timestamps do not overlap
    between Train / Validation / Test.
    """

    print("\nTemporal split validation")
    print("-" * 70)

    train_max = (
        train_df["timestamp"]
        .max()
    )

    validation_min = (
        validation_df["timestamp"]
        .min()
    )

    validation_max = (
        validation_df["timestamp"]
        .max()
    )

    test_min = (
        test_df["timestamp"]
        .min()
    )

    train_before_validation = (
        train_max
        < validation_min
    )

    validation_before_test = (
        validation_max
        < test_min
    )

    print(
        "Train ends before validation:",
        train_before_validation,
    )

    print(
        "Validation ends before test:",
        validation_before_test,
    )

    if not train_before_validation:
        raise ValueError(
            "Temporal overlap detected "
            "between Train and Validation."
        )

    if not validation_before_test:
        raise ValueError(
            "Temporal overlap detected "
            "between Validation and Test."
        )

    train_times = set(
        train_df["timestamp"]
    )

    validation_times = set(
        validation_df["timestamp"]
    )

    test_times = set(
        test_df["timestamp"]
    )

    train_validation_overlap = (
        train_times
        & validation_times
    )

    validation_test_overlap = (
        validation_times
        & test_times
    )

    train_test_overlap = (
        train_times
        & test_times
    )

    print(
        "Train/Validation timestamp overlap:",
        len(train_validation_overlap),
    )

    print(
        "Validation/Test timestamp overlap:",
        len(validation_test_overlap),
    )

    print(
        "Train/Test timestamp overlap:",
        len(train_test_overlap),
    )

    if (
        train_validation_overlap
        or validation_test_overlap
        or train_test_overlap
    ):
        raise ValueError(
            "Timestamp overlap detected "
            "between temporal splits."
        )


# ============================================================
# VALIDATE FUTURE LABEL WINDOW
# ============================================================

def validate_target_window_boundaries(
    train_df,
    validation_df,
    test_df,
    horizon_hours,
):
    """
    Verify that target windows cannot cross into
    the next temporal split.

    For Train:
        max_train_timestamp + horizon
            <
        min_validation_timestamp

    For Validation:
        max_validation_timestamp + horizon
            <
        min_test_timestamp
    """

    print(
        "\nFuture-label boundary validation"
    )

    print("-" * 70)

    horizon_delta = pd.Timedelta(
        hours=horizon_hours
    )

    train_last_timestamp = (
        train_df["timestamp"]
        .max()
    )

    validation_first_timestamp = (
        validation_df["timestamp"]
        .min()
    )

    validation_last_timestamp = (
        validation_df["timestamp"]
        .max()
    )

    test_first_timestamp = (
        test_df["timestamp"]
        .min()
    )

    train_label_end = (
        train_last_timestamp
        + horizon_delta
    )

    validation_label_end = (
        validation_last_timestamp
        + horizon_delta
    )

    train_safe = (
        train_label_end
        < validation_first_timestamp
    )

    validation_safe = (
        validation_label_end
        < test_first_timestamp
    )

    print(
        f"Last Train timestamp:        "
        f"{train_last_timestamp}"
    )

    print(
        f"Train target window ends:    "
        f"{train_label_end}"
    )

    print(
        f"Validation starts:           "
        f"{validation_first_timestamp}"
    )

    print(
        f"Train label boundary safe:   "
        f"{train_safe}"
    )

    print(
        f"\nLast Validation timestamp:   "
        f"{validation_last_timestamp}"
    )

    print(
        f"Validation target ends:      "
        f"{validation_label_end}"
    )

    print(
        f"Test starts:                 "
        f"{test_first_timestamp}"
    )

    print(
        f"Validation boundary safe:    "
        f"{validation_safe}"
    )

    if not train_safe:
        raise ValueError(
            "Future target leakage remains "
            "between Train and Validation."
        )

    if not validation_safe:
        raise ValueError(
            "Future target leakage remains "
            "between Validation and Test."
        )

    print(
        "\n[PASS] Future-label boundary "
        "validation passed."
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def print_target_distribution(
    df,
    target_column,
):
    """
    Print counts and percentages for target classes.
    """

    counts = (
        df[target_column]
        .value_counts()
        .sort_index()
    )

    percentages = (
        df[target_column]
        .value_counts(
            normalize=True
        )
        .sort_index()
        .mul(100)
        .round(2)
    )

    summary = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages,
        }
    )

    print(
        summary
    )


# ============================================================
# PRINT SPLIT SUMMARY
# ============================================================

def print_split_summary(
    train_df,
    validation_df,
    test_df,
    target_column,
    horizon_hours,
):
    """
    Print final purged split sizes,
    date ranges, and target distributions.
    """

    print("\n" + "=" * 70)

    print(
        f"PURGED TEMPORAL SPLIT SUMMARY "
        f"- {target_column}"
    )

    print("=" * 70)

    print(
        f"\nPrediction horizon: "
        f"{horizon_hours} hours"
    )

    splits = {
        "TRAIN": train_df,
        "VALIDATION": validation_df,
        "TEST": test_df,
    }

    total_rows = sum(
        len(split_df)
        for split_df in splits.values()
    )

    for name, subset in splits.items():

        percentage = (
            len(subset)
            / total_rows
            * 100
        )

        print(
            f"\n{name}"
        )

        print(
            "-" * 70
        )

        print(
            f"Rows: "
            f"{len(subset):,} "
            f"({percentage:.2f}%)"
        )

        print(
            f"Unique timestamps: "
            f"{subset['timestamp'].nunique():,}"
        )

        print(
            f"Start: "
            f"{subset['timestamp'].min()}"
        )

        print(
            f"End:   "
            f"{subset['timestamp'].max()}"
        )

        print(
            "\nTarget distribution:"
        )

        print_target_distribution(
            subset,
            target_column,
        )


# ============================================================
# BUILD DATASET FOR ONE HORIZON
# ============================================================

def build_horizon_dataset(
    modeling_df,
    feature_columns,
    target_column,
):
    """
    Build one horizon-specific purged temporal dataset.
    """

    if target_column not in TARGET_HORIZONS:
        raise ValueError(
            f"No horizon configured for "
            f"{target_column}"
        )

    horizon_hours = (
        TARGET_HORIZONS[
            target_column
        ]
    )

    print("\n" + "#" * 70)

    print(
        f"BUILDING DATASET: "
        f"{target_column}"
    )

    print("#" * 70)

    print(
        f"Prediction horizon: "
        f"{horizon_hours} hours"
    )

    # --------------------------------------------------------
    # Remove rows where this target is unavailable
    # --------------------------------------------------------

    rows_before = len(
        modeling_df
    )

    horizon_df = (
        modeling_df
        .dropna(
            subset=[
                target_column
            ]
        )
        .copy()
    )

    rows_after = len(
        horizon_df
    )

    print(
        "\nRows removed because target "
        f"is unavailable: "
        f"{rows_before - rows_after:,}"
    )

    print(
        f"Rows available for "
        f"{target_column}: "
        f"{rows_after:,}"
    )

    # --------------------------------------------------------
    # Select only model-relevant columns
    # --------------------------------------------------------

    identifier_columns = [
        "coin_id",
        "timestamp",
    ]

    if "symbol" in horizon_df.columns:

        identifier_columns.insert(
            1,
            "symbol",
        )

    selected_columns = (
        identifier_columns
        + list(feature_columns)
        + [target_column]
    )

    horizon_df = (
        horizon_df[
            selected_columns
        ]
        .copy()
        .sort_values(
            [
                "timestamp",
                "coin_id",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Final missing checks
    # --------------------------------------------------------

    feature_nulls = (
        horizon_df[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    target_nulls = (
        horizon_df[
            target_column
        ]
        .isna()
        .sum()
    )

    if feature_nulls > 0:
        raise ValueError(
            f"{feature_nulls} feature "
            "null values remain."
        )

    if target_nulls > 0:
        raise ValueError(
            f"{target_nulls} target "
            "null values remain."
        )

    # --------------------------------------------------------
    # Initial chronological split
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = create_initial_temporal_split(
        horizon_df
    )

    print(
        "\nInitial split before purge"
    )

    print(
        "-" * 70
    )

    print(
        f"Train rows:      "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Test rows:       "
        f"{len(test_df):,}"
    )

    # --------------------------------------------------------
    # Apply horizon-specific purge
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = apply_horizon_purge(
        train_df,
        validation_df,
        test_df,
        horizon_hours,
    )

    # --------------------------------------------------------
    # Validate ordinary timestamp separation
    # --------------------------------------------------------

    validate_temporal_order(
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # Validate future label windows
    # --------------------------------------------------------

    validate_target_window_boundaries(
        train_df,
        validation_df,
        test_df,
        horizon_hours,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_split_summary(
        train_df,
        validation_df,
        test_df,
        target_column,
        horizon_hours,
    )

    return {
        "full": horizon_df,
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
        "horizon_hours": horizon_hours,
    }


# ============================================================
# BUILD ALL TEMPORAL DATASETS
# ============================================================

def build_temporal_datasets(
    df,
    feature_columns,
):
    """
    Build purged temporal datasets for:

        target_6h
        target_12h
        target_24h
    """

    validate_split_ratios()

    modeling_df = (
        prepare_modeling_dataframe(
            df,
            feature_columns,
        )
    )

    datasets = {}

    for target_column in TARGET_COLUMNS:

        datasets[target_column] = (
            build_horizon_dataset(
                modeling_df,
                feature_columns,
                target_column,
            )
        )

    print("\n" + "=" * 70)

    print(
        "PURGED TEMPORAL MODELING "
        "DATASET COMPLETE"
    )

    print("=" * 70)

    print(
        "\nPrediction horizons:"
    )

    for target_column in TARGET_COLUMNS:

        horizon_hours = (
            TARGET_HORIZONS[
                target_column
            ]
        )

        print(
            f"[✓] {target_column} "
            f"with {horizon_hours}h purge"
        )

    print(
        "\n[✓] Chronological splitting"
    )

    print(
        "[✓] No random shuffle"
    )

    print(
        "[✓] No timestamp overlap"
    )

    print(
        "[✓] Train target windows "
        "cannot enter Validation"
    )

    print(
        "[✓] Validation target windows "
        "cannot enter Test"
    )

    print(
        "[✓] Horizon-specific purge"
    )

    print(
        "[✓] No CSV files created"
    )

    print(
        "[✓] PostgreSQL remains "
        "source of truth"
    )

    return datasets