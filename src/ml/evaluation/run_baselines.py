"""
Baseline Models
===============

This module evaluates simple baseline classifiers for the
cryptocurrency trend-prediction project.

Current baseline:
- Majority-class classifier

Important:
- Uses the already validated temporal datasets.
- Train / Validation / Test remain chronological.
- Horizon-specific purge has already been applied.
- Majority class is learned ONLY from the training split.
- Validation is used for baseline evaluation.
- Test remains untouched until final model evaluation.
- PostgreSQL remains the source of truth.
- No CSV files are used.
"""

from collections import Counter

import numpy as np
import pandas as pd

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


# ============================================================
# MAJORITY CLASS
# ============================================================

def get_majority_class(
    y_train,
):
    """
    Find the most common target class in the training split.

    The majority class must come ONLY from training data.
    Validation and Test must never influence this decision.
    """

    if len(y_train) == 0:
        raise ValueError(
            "Training target is empty."
        )

    class_counts = Counter(
        y_train
    )

    majority_class = (
        class_counts
        .most_common(1)[0][0]
    )

    return majority_class


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def print_class_distribution(
    y,
    name,
):
    """
    Print class counts and percentages.
    """

    print(
        f"\n{name} target distribution"
    )

    print("-" * 70)

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
        distribution
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Print confusion matrix with readable labels.
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

    print("-" * 70)

    print(
        matrix_df
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Calculate baseline classification metrics.

    Accuracy alone is not enough because the target
    classes are not perfectly balanced.

    Therefore we also calculate:
    - Macro F1
    - Balanced Accuracy
    - Macro Precision
    - Macro Recall
    """

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "macro_precision":
            precision_score(
                y_true,
                y_pred,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            ),

        "macro_recall":
            recall_score(
                y_true,
                y_pred,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            ),
    }

    return metrics


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    metrics,
):
    """
    Print evaluation metrics clearly.
    """

    print(
        "\nBaseline metrics"
    )

    print("-" * 70)

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

    print("-" * 70)

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        target_names=CLASS_ORDER,
        digits=4,
        zero_division=0,
    )

    print(
        report
    )


# ============================================================
# EVALUATE ONE HORIZON
# ============================================================

def evaluate_majority_baseline(
    dataset,
    target_column,
):
    """
    Evaluate majority-class baseline for one horizon.

    Training split:
        Used to determine majority class.

    Validation split:
        Used to evaluate baseline performance.

    Test split:
        NOT evaluated.
    """

    print("\n" + "=" * 70)

    print(
        f"MAJORITY BASELINE - "
        f"{target_column}"
    )

    print("=" * 70)

    train_df = dataset[
        "train"
    ]

    validation_df = dataset[
        "validation"
    ]

    test_df = dataset[
        "test"
    ]

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if train_df.empty:
        raise ValueError(
            f"{target_column}: "
            "training dataset is empty."
        )

    if validation_df.empty:
        raise ValueError(
            f"{target_column}: "
            "validation dataset is empty."
        )

    if test_df.empty:
        raise ValueError(
            f"{target_column}: "
            "test dataset is empty."
        )

    # --------------------------------------------------------
    # Extract targets
    # --------------------------------------------------------

    y_train = (
        train_df[
            target_column
        ]
        .astype(str)
    )

    y_validation = (
        validation_df[
            target_column
        ]
        .astype(str)
    )

    # --------------------------------------------------------
    # Show distributions
    # --------------------------------------------------------

    print_class_distribution(
        y_train,
        "TRAIN",
    )

    print_class_distribution(
        y_validation,
        "VALIDATION",
    )

    # --------------------------------------------------------
    # Determine majority class using TRAIN ONLY
    # --------------------------------------------------------

    majority_class = (
        get_majority_class(
            y_train
        )
    )

    print(
        "\nMajority class learned "
        "from training data"
    )

    print("-" * 70)

    print(
        majority_class
    )

    # --------------------------------------------------------
    # Predict same class for every validation row
    # --------------------------------------------------------

    y_prediction = np.full(
        shape=len(
            y_validation
        ),
        fill_value=majority_class,
        dtype=object,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_validation,
        y_prediction,
    )

    print_metrics(
        metrics
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print_confusion_matrix(
        y_validation,
        y_prediction,
    )

    # --------------------------------------------------------
    # Per-class report
    # --------------------------------------------------------

    print_classification_report(
        y_validation,
        y_prediction,
    )

    # --------------------------------------------------------
    # Confirm Test remains untouched
    # --------------------------------------------------------

    print(
        "\nTest-set status"
    )

    print("-" * 70)

    print(
        f"Test rows available: "
        f"{len(test_df):,}"
    )

    print(
        "Test evaluation performed: NO"
    )

    print(
        "[✓] Test set remains untouched."
    )

    return {
        "target_column":
            target_column,

        "majority_class":
            majority_class,

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
    }


# ============================================================
# SUMMARY
# ============================================================

def print_baseline_summary(
    results,
):
    """
    Print comparison across all prediction horizons.
    """

    print("\n" + "=" * 80)

    print(
        "MAJORITY-CLASS BASELINE SUMMARY"
    )

    print("=" * 80)

    summary_df = pd.DataFrame(
        results
    )

    summary_df = summary_df[
        [
            "target_column",
            "majority_class",
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ]
    ]

    print(
        "\nValidation results"
    )

    print("-" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nInterpretation"
    )

    print("-" * 80)

    print(
        "These scores are the minimum benchmark "
        "that our trained ML models should beat."
    )

    print(
        "Because the majority classifier predicts "
        "only one class, balanced accuracy and "
        "macro F1 are more informative than "
        "accuracy alone."
    )

    print(
        "\n[✓] Majority baseline complete"
    )

    print(
        "[✓] Validation used for evaluation"
    )

    print(
        "[✓] Test set untouched"
    )

    return summary_df


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run temporal dataset construction and evaluate
    majority-class baselines for all horizons.
    """

    print("\n" + "=" * 80)

    print(
        "RUNNING BASELINE MODELING PIPELINE"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Run existing DB -> EDA -> Feature -> Temporal pipeline
    #
    # analyze_returns.main() now returns:
    #
    # {
    #     "target_6h": {...},
    #     "target_12h": {...},
    #     "target_24h": {...},
    # }
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

    results = []

    # --------------------------------------------------------
    # Evaluate each prediction horizon separately
    # --------------------------------------------------------

    for target_column in TARGET_COLUMNS:

        if target_column not in temporal_datasets:
            raise KeyError(
                f"{target_column} was not found "
                "in temporal datasets."
            )

        result = (
            evaluate_majority_baseline(
                temporal_datasets[
                    target_column
                ],
                target_column,
            )
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    summary_df = (
        print_baseline_summary(
            results
        )
    )

    return summary_df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()