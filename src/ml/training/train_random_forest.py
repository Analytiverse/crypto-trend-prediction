"""
Random Forest Modeling
======================

Train and evaluate Random Forest models for:

- target_6h
- target_12h
- target_24h

Two variants are compared:

1. Standard Random Forest
2. Balanced Random Forest using class_weight="balanced"

Important:
- PostgreSQL remains the source of truth.
- Existing feature engineering is reused.
- Existing purged temporal splits are reused.
- Train is used for fitting.
- Validation is used for model comparison.
- Test remains untouched.
- No CSV files are used.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.analysis.analyze_returns import main as run_analysis_pipeline


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMNS = [
    "target_6h",
    "target_12h",
    "target_24h",
]

CLASS_ORDER = [
    "DOWN",
    "STABLE",
    "UP",
]


FEATURE_COLUMNS = [
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


# Best Logistic Regression validation Macro F1 scores
# from the previous modeling phase.

LOGISTIC_MACRO_F1 = {
    "target_6h": 0.449475,
    "target_12h": 0.513312,
    "target_24h": 0.512767,
}


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Calculate classification metrics.
    """

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),

        "macro_precision": precision_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),

        "macro_recall": recall_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    metrics,
):
    """
    Print metrics clearly.
    """

    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro Precision:   "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall:      "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{metrics['macro_f1']:.4f}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Print readable confusion matrix.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"actual_{label}"
            for label in CLASS_ORDER
        ],
        columns=[
            f"predicted_{label}"
            for label in CLASS_ORDER
        ],
    )

    print(
        "\nConfusion matrix"
    )

    print(
        "-" * 70
    )

    print(
        matrix_df
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def print_classification_report(
    y_true,
    y_pred,
):
    """
    Print precision, recall and F1 for each class.
    """

    print(
        "\nPer-class classification report"
    )

    print(
        "-" * 70
    )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            target_names=CLASS_ORDER,
            digits=4,
            zero_division=0,
        )
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def print_target_distribution(
    y,
    name,
):
    """
    Print class distribution.
    """

    counts = (
        pd.Series(y)
        .value_counts()
        .reindex(
            CLASS_ORDER,
            fill_value=0,
        )
    )

    percentages = (
        counts
        / counts.sum()
        * 100
    ).round(2)

    distribution = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages,
        }
    )

    print(
        f"\n{name} target distribution"
    )

    print(
        "-" * 70
    )

    print(
        distribution
    )


# ============================================================
# BUILD RANDOM FOREST
# ============================================================

def build_random_forest(
    balanced=False,
):
    """
    Build Random Forest classifier.

    No StandardScaler is required because tree-based models
    are not sensitive to feature scale.
    """

    if balanced:

        class_weight = "balanced"

    else:

        class_weight = None

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )

    return model


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def print_feature_importance(
    model,
):
    """
    Print Random Forest feature importance.
    """

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nFeature importance"
    )

    print(
        "-" * 70
    )

    print(
        importance_df.to_string(
            index=False
        )
    )

    return importance_df


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model_variant(
    model_name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    target_column,
):
    """
    Train one Random Forest and evaluate on Validation.
    """

    print("\n" + "-" * 70)

    print(
        model_name
    )

    print(
        "-" * 70
    )

    print(
        f"Training rows:   "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_validation):,}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Predict Validation
    # --------------------------------------------------------

    predictions = (
        model.predict(
            X_validation
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_validation,
        predictions,
    )

    print(
        "\nValidation metrics"
    )

    print(
        "-" * 70
    )

    print_metrics(
        metrics
    )

    # --------------------------------------------------------
    # Compare against Logistic Regression
    # --------------------------------------------------------

    logistic_f1 = (
        LOGISTIC_MACRO_F1[
            target_column
        ]
    )

    difference = (
        metrics["macro_f1"]
        - logistic_f1
    )

    print(
        "\nComparison with Logistic Regression"
    )

    print(
        "-" * 70
    )

    print(
        f"Logistic Macro F1: "
        f"{logistic_f1:.4f}"
    )

    print(
        f"Random Forest Macro F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Difference: "
        f"{difference:+.4f}"
    )

    if difference > 0:

        print(
            "[PASS] Random Forest beats Logistic Regression."
        )

    elif difference == 0:

        print(
            "[TIE] Random Forest matches Logistic Regression."
        )

    else:

        print(
            "[INFO] Random Forest does not beat Logistic Regression."
        )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    print_target_distribution(
        predictions,
        "PREDICTED VALIDATION",
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print_confusion_matrix(
        y_validation,
        predictions,
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print_classification_report(
        y_validation,
        predictions,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance_df = (
        print_feature_importance(
            model
        )
    )

    return {
        "target_column":
            target_column,

        "model":
            model_name,

        "accuracy":
            metrics[
                "accuracy"
            ],

        "balanced_accuracy":
            metrics[
                "balanced_accuracy"
            ],

        "macro_precision":
            metrics[
                "macro_precision"
            ],

        "macro_recall":
            metrics[
                "macro_recall"
            ],

        "macro_f1":
            metrics[
                "macro_f1"
            ],

        "logistic_macro_f1":
            logistic_f1,

        "difference_vs_logistic":
            difference,

        "trained_model":
            model,

        "feature_importance":
            importance_df,
    }


# ============================================================
# TRAIN ONE HORIZON
# ============================================================

def train_horizon_models(
    dataset,
    target_column,
):
    """
    Train Standard and Balanced Random Forest
    for one prediction horizon.
    """

    print("\n" + "=" * 80)

    print(
        f"RANDOM FOREST - "
        f"{target_column}"
    )

    print("=" * 80)

    train_df = (
        dataset[
            "train"
        ]
        .copy()
    )

    validation_df = (
        dataset[
            "validation"
        ]
        .copy()
    )

    test_df = (
        dataset[
            "test"
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Required feature check
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in train_df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing features for "
            f"{target_column}: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Extract X / y
    # --------------------------------------------------------

    X_train = (
        train_df[
            FEATURE_COLUMNS
        ]
        .copy()
    )

    y_train = (
        train_df[
            target_column
        ]
        .astype(str)
        .copy()
    )

    X_validation = (
        validation_df[
            FEATURE_COLUMNS
        ]
        .copy()
    )

    y_validation = (
        validation_df[
            target_column
        ]
        .astype(str)
        .copy()
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if X_train.isna().any().any():

        raise ValueError(
            f"{target_column}: "
            "Training features contain missing values."
        )

    if X_validation.isna().any().any():

        raise ValueError(
            f"{target_column}: "
            "Validation features contain missing values."
        )

    if not np.isfinite(
        X_train.to_numpy()
    ).all():

        raise ValueError(
            f"{target_column}: "
            "Training features contain infinite values."
        )

    if not np.isfinite(
        X_validation.to_numpy()
    ).all():

        raise ValueError(
            f"{target_column}: "
            "Validation features contain infinite values."
        )

    # --------------------------------------------------------
    # Dataset info
    # --------------------------------------------------------

    print(
        f"\nFeatures used: "
        f"{len(FEATURE_COLUMNS)}"
    )

    for feature in FEATURE_COLUMNS:

        print(
            f"  - {feature}"
        )

    print_target_distribution(
        y_train,
        "TRAIN",
    )

    print_target_distribution(
        y_validation,
        "VALIDATION",
    )

    # --------------------------------------------------------
    # Standard Random Forest
    # --------------------------------------------------------

    standard_model = (
        build_random_forest(
            balanced=False
        )
    )

    standard_result = (
        evaluate_model_variant(
            model_name=(
                "Standard Random Forest"
            ),
            model=standard_model,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            target_column=target_column,
        )
    )

    # --------------------------------------------------------
    # Balanced Random Forest
    # --------------------------------------------------------

    balanced_model = (
        build_random_forest(
            balanced=True
        )
    )

    balanced_result = (
        evaluate_model_variant(
            model_name=(
                "Balanced Random Forest"
            ),
            model=balanced_model,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            target_column=target_column,
        )
    )

    # --------------------------------------------------------
    # Select best Random Forest
    # --------------------------------------------------------

    if (
        balanced_result["macro_f1"]
        >
        standard_result["macro_f1"]
    ):

        best_result = (
            balanced_result
        )

    else:

        best_result = (
            standard_result
        )

    print("\n" + "=" * 70)

    print(
        f"BEST RANDOM FOREST - "
        f"{target_column}"
    )

    print("=" * 70)

    print(
        f"Selected model: "
        f"{best_result['model']}"
    )

    print(
        f"Validation Macro F1: "
        f"{best_result['macro_f1']:.4f}"
    )

    print(
        f"Logistic Regression Macro F1: "
        f"{best_result['logistic_macro_f1']:.4f}"
    )

    print(
        f"Difference: "
        f"{best_result['difference_vs_logistic']:+.4f}"
    )

    print(
        "\nTest-set status"
    )

    print(
        "-" * 70
    )

    print(
        f"Test rows available: "
        f"{len(test_df):,}"
    )

    print(
        "Test predictions generated: NO"
    )

    print(
        "Test evaluation performed: NO"
    )

    print(
        "[✓] Test set remains untouched."
    )

    return {
        "standard":
            standard_result,

        "balanced":
            balanced_result,

        "best":
            best_result,
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    all_results,
):
    """
    Print best Random Forest results
    across all horizons.
    """

    summary_rows = []

    for (
        target_column,
        results,
    ) in all_results.items():

        best = (
            results[
                "best"
            ]
        )

        summary_rows.append(
            {
                "target_column":
                    target_column,

                "selected_model":
                    best[
                        "model"
                    ],

                "accuracy":
                    best[
                        "accuracy"
                    ],

                "balanced_accuracy":
                    best[
                        "balanced_accuracy"
                    ],

                "macro_precision":
                    best[
                        "macro_precision"
                    ],

                "macro_recall":
                    best[
                        "macro_recall"
                    ],

                "macro_f1":
                    best[
                        "macro_f1"
                    ],

                "logistic_macro_f1":
                    best[
                        "logistic_macro_f1"
                    ],

                "difference":
                    best[
                        "difference_vs_logistic"
                    ],
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    print("\n" + "=" * 100)

    print(
        "RANDOM FOREST SUMMARY"
    )

    print("=" * 100)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)

    print(
        "RANDOM FOREST PHASE COMPLETE"
    )

    print("=" * 100)

    print(
        "\n[✓] Standard Random Forest trained"
    )

    print(
        "[✓] Balanced Random Forest trained"
    )

    print(
        "[✓] Validation used for model comparison"
    )

    print(
        "[✓] Macro F1 used for model selection"
    )

    print(
        "[✓] Feature importance calculated"
    )

    print(
        "[✓] Logistic Regression comparison completed"
    )

    print(
        "[✓] Test sets remain untouched"
    )

    return summary_df


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run complete Random Forest modeling phase.
    """

    print("\n" + "=" * 100)

    print(
        "RUNNING RANDOM FOREST MODELING PIPELINE"
    )

    print("=" * 100)

    # --------------------------------------------------------
    # Existing pipeline:
    #
    # PostgreSQL
    #     ↓
    # EDA
    #     ↓
    # Targets
    #     ↓
    # Features
    #     ↓
    # Leakage validation
    #     ↓
    # Purged temporal datasets
    #
    # --------------------------------------------------------

    temporal_datasets = (
        run_analysis_pipeline()
    )

    if temporal_datasets is None:

        raise ValueError(
            "Temporal datasets were not returned "
            "by analyze_returns.main()."
        )

    all_results = {}

    # --------------------------------------------------------
    # Train all horizons
    # --------------------------------------------------------

    for target_column in TARGET_COLUMNS:

        if (
            target_column
            not in temporal_datasets
        ):

            raise KeyError(
                f"{target_column} "
                "was not found in temporal datasets."
            )

        results = (
            train_horizon_models(
                temporal_datasets[
                    target_column
                ],
                target_column,
            )
        )

        all_results[
            target_column
        ] = results

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_df = (
        print_final_summary(
            all_results
        )
    )

    return {
        "results":
            all_results,

        "summary":
            summary_df,
    }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()