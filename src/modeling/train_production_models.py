"""
Train the final production models for AlphaPulse.

The model family has ALREADY been selected using validation
performance and evaluated once on the final test set.

Selected model:
    Balanced Logistic Regression

This script does NOT perform model selection.

It refits the already-selected model architecture using the
currently available labeled historical dataset so that the
resulting artifacts can be used by the production inference
system.

Outputs:
    artifacts/models/model_6h.joblib
    artifacts/models/model_12h.joblib
    artifacts/models/model_24h.joblib

    artifacts/metadata/model_6h.json
    artifacts/metadata/model_12h.json
    artifacts/metadata/model_24h.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.features.feature_builder import (
    ML_FEATURES,
    build_features,
    load_all_market_data,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "metadata"
)

HORIZON_THRESHOLDS = {
    6: 0.005,
    12: 0.010,
    24: 0.015,
}

CLASS_NAMES = [
    "DOWN",
    "STABLE",
    "UP",
]


# ============================================================
# FUTURE RETURN CONSTRUCTION
# ============================================================

def create_future_return(
    df: pd.DataFrame,
    horizon_hours: int,
) -> pd.DataFrame:
    """
    Calculate future return using exact timestamp alignment.

    future_return_h =
        price(t+h) / price(t) - 1
    """

    df = df.copy()

    future_prices = df[
        [
            "coin_id",
            "timestamp",
            "price",
        ]
    ].copy()

    future_prices["timestamp"] = (
        future_prices["timestamp"]
        - pd.Timedelta(
            hours=horizon_hours
        )
    )

    future_price_column = (
        f"price_future_{horizon_hours}h"
    )

    future_prices = (
        future_prices.rename(
            columns={
                "price":
                    future_price_column
            }
        )
    )

    df = df.merge(
        future_prices,
        on=[
            "coin_id",
            "timestamp",
        ],
        how="left",
    )

    future_return_column = (
        f"future_return_{horizon_hours}h"
    )

    df[future_return_column] = (
        df[future_price_column]
        / df["price"]
        - 1
    )

    return df


# ============================================================
# TARGET CONSTRUCTION
# ============================================================

def create_target(
    df: pd.DataFrame,
    horizon_hours: int,
    threshold: float,
) -> pd.DataFrame:
    """
    Create DOWN / STABLE / UP target.

    return > +threshold  -> UP
    return < -threshold  -> DOWN
    otherwise            -> STABLE
    """

    df = create_future_return(
        df,
        horizon_hours,
    )

    return_column = (
        f"future_return_{horizon_hours}h"
    )

    target_column = (
        f"target_{horizon_hours}h"
    )

    df[target_column] = pd.NA

    valid = df[
        return_column
    ].notna()

    df.loc[
        valid,
        target_column,
    ] = "STABLE"

    df.loc[
        valid
        &
        (
            df[return_column]
            > threshold
        ),
        target_column,
    ] = "UP"

    df.loc[
        valid
        &
        (
            df[return_column]
            < -threshold
        ),
        target_column,
    ] = "DOWN"

    return df


# ============================================================
# CREATE MODEL
# ============================================================

def create_model() -> Pipeline:
    """
    Create the already-selected final model architecture.

    Saving the scaler and classifier inside one sklearn
    Pipeline prevents training-serving preprocessing mismatch.
    """

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# TRAIN ONE HORIZON
# ============================================================

def train_horizon(
    feature_df: pd.DataFrame,
    horizon_hours: int,
) -> dict:
    """
    Train and save one horizon-specific production model.
    """

    threshold = (
        HORIZON_THRESHOLDS[
            horizon_hours
        ]
    )

    print()
    print("=" * 70)
    print(
        f"TRAINING PRODUCTION MODEL - "
        f"{horizon_hours}H"
    )
    print("=" * 70)

    df = create_target(
        feature_df,
        horizon_hours,
        threshold,
    )

    target_column = (
        f"target_{horizon_hours}h"
    )

    required_columns = (
        ML_FEATURES
        + [target_column]
    )

    modeling_df = (
        df.dropna(
            subset=required_columns
        )
        .copy()
    )

    if modeling_df.empty:
        raise ValueError(
            f"No training rows available for "
            f"{horizon_hours}h."
        )

    X = modeling_df[
        ML_FEATURES
    ].copy()

    y = (
        modeling_df[
            target_column
        ]
        .astype(str)
    )

    print(
        f"Training rows: "
        f"{len(modeling_df):,}"
    )

    print()
    print("Target distribution:")
    print(
        y.value_counts(
            normalize=False
        )
    )

    print()
    print("Target percentages:")
    print(
        (
            y.value_counts(
                normalize=True
            )
            * 100
        ).round(2)
    )

    model = create_model()

    model.fit(
        X,
        y,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / f"model_{horizon_hours}h.joblib"
    )

    metadata_path = (
        METADATA_DIR
        / f"model_{horizon_hours}h.json"
    )

    joblib.dump(
        model,
        model_path,
    )

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    metadata = {
        "project": "AlphaPulse",
        "model_version": "1.0",
        "model_name":
            "Balanced Logistic Regression",
        "horizon_hours":
            horizon_hours,
        "threshold":
            threshold,
        "features":
            ML_FEATURES,
        "classes":
            classifier.classes_.tolist(),
        "training_rows":
            int(len(modeling_df)),
        "training_start":
            modeling_df[
                "timestamp"
            ].min().isoformat(),
        "training_end":
            modeling_df[
                "timestamp"
            ].max().isoformat(),
        "coins":
            sorted(
                modeling_df[
                    "coin_id"
                ]
                .unique()
                .tolist()
            ),
        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "notes": (
            "Production refit of the already-selected "
            "Balanced Logistic Regression architecture. "
            "The previous held-out test evaluation remains "
            "the final historical model-selection evaluation."
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print()
    print(
        f"[✓] Model saved: "
        f"{model_path}"
    )

    print(
        f"[✓] Metadata saved: "
        f"{metadata_path}"
    )

    return metadata


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print(
        "ALPHAPULSE - PRODUCTION MODEL TRAINING"
    )
    print("=" * 70)

    print()
    print(
        "Loading cleaned hourly market data "
        "from PostgreSQL..."
    )

    raw_df = load_all_market_data()

    print(
        f"Loaded rows: {len(raw_df):,}"
    )

    print(
        f"Date range: "
        f"{raw_df['timestamp'].min()} "
        f"-> "
        f"{raw_df['timestamp'].max()}"
    )

    print()
    print(
        "Building leakage-safe historical features..."
    )

    feature_df = build_features(
        raw_df
    )

    print(
        f"Feature rows built: "
        f"{len(feature_df):,}"
    )

    summaries = []

    for horizon_hours in [
        6,
        12,
        24,
    ]:

        metadata = train_horizon(
            feature_df,
            horizon_hours,
        )

        summaries.append(
            metadata
        )

    print()
    print("=" * 70)
    print(
        "PRODUCTION MODEL EXPORT COMPLETE"
    )
    print("=" * 70)

    for metadata in summaries:

        print(
            f"[✓] "
            f"{metadata['horizon_hours']}h "
            f"-> "
            f"{metadata['model_name']}"
        )

    print()
    print(
        "Created production artifacts:"
    )

    print(
        "artifacts/models/model_6h.joblib"
    )

    print(
        "artifacts/models/model_12h.joblib"
    )

    print(
        "artifacts/models/model_24h.joblib"
    )


if __name__ == "__main__":
    main()