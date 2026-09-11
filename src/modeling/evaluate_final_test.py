"""
Final Test Evaluation
=====================

Perform the one-time final evaluation of the selected model.

Selected model:
    Balanced Logistic Regression

Prediction horizons:
    - target_6h
    - target_12h
    - target_24h

Methodology:
    1. Rebuild the existing purged temporal datasets.
    2. Combine Train + Validation into the final development set.
    3. Fit StandardScaler on Train + Validation only.
    4. Train Balanced Logistic Regression on Train + Validation only.
    5. Evaluate exactly once on Test.
    6. Compare against a majority-class baseline.

Important:
    - PostgreSQL remains the source of truth.
    - No CSV modeling dataset is used.
    - Test is never used for fitting or preprocessing.
    - Test results must NOT be used to redesign or retune the model.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


# Validation Macro F1 from final model selection.
#
# These values are NOT used to tune the final model.
# They are only printed later for comparison with Test performance.

VALIDATION_MACRO_F1 = {
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
    Calculate final classification metrics.
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
    Print evaluation metrics.
    """

    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"Macro Precision:   "
        f"{metrics['macro_precision']:.6f}"
    )

    print(
        f"Macro Recall:      "
        f"{metrics['macro_recall']:.6f}"
    )

    print(
        f"Macro F1:          "
        f"{metrics['macro_f1']:.6f}"
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
# CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Print final confusion matrix.
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
    Print per-class Test metrics.
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
# BUILD FINAL MODEL
# ============================================================

def build_final_model():
    """
    Build the selected final model.

    Model:
        Balanced Logistic Regression

    StandardScaler is included inside the Pipeline so it is
    fitted only using the development dataset passed to fit().
    """

    model = Pipeline(
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
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    return model


# ============================================================
# MAJORITY BASELINE
# ============================================================

def evaluate_majority_baseline(
    y_development,
    y_test,
):
    """
    Learn the majority class using Train + Validation only.

    Predict that class for every Test row.

    The Test labels are not used to choose the majority class.
    """

    majority_class = (
        pd.Series(
            y_development
        )
        .value_counts()
        .idxmax()
    )

    predictions = np.full(
        shape=len(y_test),
        fill_value=majority_class,
        dtype=object,
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    return {
        "majority_class":
            majority_class,

        "predictions":
            predictions,

        "metrics":
            metrics,
    }


# ============================================================
# FINAL EVALUATION FOR ONE HORIZON
# ============================================================

def evaluate_horizon(
    dataset,
    target_column,
):
    """
    Refit selected model on Train + Validation
    and evaluate exactly once on Test.
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"FINAL TEST EVALUATION - "
        f"{target_column}"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Load split DataFrames
    # --------------------------------------------------------

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
    # Combine Train + Validation
    #
    # Model selection is already finished.
    #
    # We can now use all available development history
    # before the Test period to fit the selected model.
    # --------------------------------------------------------

    development_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        axis=0,
        ignore_index=True,
    )

    development_df = (
        development_df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Feature checks
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in development_df.columns
    ]

    if missing_features:

        raise ValueError(
            f"{target_column}: "
            f"Missing features: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Build X / y
    # --------------------------------------------------------

    X_development = (
        development_df[
            FEATURE_COLUMNS
        ]
        .copy()
    )

    y_development = (
        development_df[
            target_column
        ]
        .astype(str)
        .copy()
    )

    X_test = (
        test_df[
            FEATURE_COLUMNS
        ]
        .copy()
    )

    y_test = (
        test_df[
            target_column
        ]
        .astype(str)
        .copy()
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if X_development.isna().any().any():

        raise ValueError(
            f"{target_column}: "
            "Development features contain missing values."
        )

    if X_test.isna().any().any():

        raise ValueError(
            f"{target_column}: "
            "Test features contain missing values."
        )

    if not np.isfinite(
        X_development.to_numpy()
    ).all():

        raise ValueError(
            f"{target_column}: "
            "Development features contain infinite values."
        )

    if not np.isfinite(
        X_test.to_numpy()
    ).all():

        raise ValueError(
            f"{target_column}: "
            "Test features contain infinite values."
        )

    # --------------------------------------------------------
    # Temporal safety checks
    # --------------------------------------------------------

    development_end = (
        development_df[
            "timestamp"
        ]
        .max()
    )

    test_start = (
        test_df[
            "timestamp"
        ]
        .min()
    )

    chronology_safe = (
        development_end
        <
        test_start
    )

    print(
        "\nDataset sizes"
    )

    print(
        "-" * 70
    )

    print(
        f"Original Train rows:      "
        f"{len(train_df):,}"
    )

    print(
        f"Original Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Final development rows:   "
        f"{len(development_df):,}"
    )

    print(
        f"Final Test rows:          "
        f"{len(test_df):,}"
    )

    print(
        "\nTemporal boundary"
    )

    print(
        "-" * 70
    )

    print(
        f"Development ends: "
        f"{development_end}"
    )

    print(
        f"Test starts:      "
        f"{test_start}"
    )

    print(
        f"Development ends before Test: "
        f"{chronology_safe}"
    )

    if not chronology_safe:

        raise ValueError(
            f"{target_column}: "
            "Development/Test chronology is invalid."
        )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print_target_distribution(
        y_development,
        "FINAL DEVELOPMENT",
    )

    print_target_distribution(
        y_test,
        "TEST",
    )

    # --------------------------------------------------------
    # Majority baseline
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )

    print(
        "TEST MAJORITY BASELINE"
    )

    print(
        "-" * 70
    )

    baseline_result = (
        evaluate_majority_baseline(
            y_development,
            y_test,
        )
    )

    print(
        f"Majority class learned from "
        f"development data: "
        f"{baseline_result['majority_class']}"
    )

    print(
        "\nBaseline Test metrics"
    )

    print(
        "-" * 70
    )

    print_metrics(
        baseline_result[
            "metrics"
        ]
    )

    # --------------------------------------------------------
    # Final Balanced Logistic Regression
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )

    print(
        "FINAL BALANCED LOGISTIC REGRESSION"
    )

    print(
        "-" * 70
    )

    final_model = (
        build_final_model()
    )

    print(
        f"Training final model on "
        f"{len(X_development):,} "
        f"Train + Validation rows."
    )

    # --------------------------------------------------------
    # Fit final model
    #
    # StandardScaler and Logistic Regression are fitted
    # using development data only.
    # --------------------------------------------------------

    final_model.fit(
        X_development,
        y_development,
    )

    # --------------------------------------------------------
    # ONE-TIME TEST PREDICTION
    # --------------------------------------------------------

    test_predictions = (
        final_model.predict(
            X_test
        )
    )

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    test_metrics = (
        calculate_metrics(
            y_test,
            test_predictions,
        )
    )

    print(
        "\nFinal Test metrics"
    )

    print(
        "-" * 70
    )

    print_metrics(
        test_metrics
    )

    # --------------------------------------------------------
    # Compare with baseline
    # --------------------------------------------------------

    baseline_macro_f1 = (
        baseline_result[
            "metrics"
        ][
            "macro_f1"
        ]
    )

    improvement_vs_baseline = (
        test_metrics[
            "macro_f1"
        ]
        -
        baseline_macro_f1
    )

    print(
        "\nComparison with Test majority baseline"
    )

    print(
        "-" * 70
    )

    print(
        f"Baseline Test Macro F1: "
        f"{baseline_macro_f1:.6f}"
    )

    print(
        f"Final model Test Macro F1: "
        f"{test_metrics['macro_f1']:.6f}"
    )

    print(
        f"Absolute improvement: "
        f"{improvement_vs_baseline:+.6f}"
    )

    # --------------------------------------------------------
    # Compare Validation and Test
    #
    # This is diagnostic only.
    #
    # We must NOT retune the model based on this difference.
    # --------------------------------------------------------

    validation_f1 = (
        VALIDATION_MACRO_F1[
            target_column
        ]
    )

    validation_test_difference = (
        test_metrics[
            "macro_f1"
        ]
        -
        validation_f1
    )

    print(
        "\nValidation vs Test"
    )

    print(
        "-" * 70
    )

    print(
        f"Selected Validation Macro F1: "
        f"{validation_f1:.6f}"
    )

    print(
        f"Final Test Macro F1:           "
        f"{test_metrics['macro_f1']:.6f}"
    )

    print(
        f"Difference:                    "
        f"{validation_test_difference:+.6f}"
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    print_target_distribution(
        test_predictions,
        "PREDICTED TEST",
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print_confusion_matrix(
        y_test,
        test_predictions,
    )

    # --------------------------------------------------------
    # Per-class report
    # --------------------------------------------------------

    print_classification_report(
        y_test,
        test_predictions,
    )

    print(
        "\n[✓] Final Test evaluation completed."
    )

    print(
        "[✓] Test data was not used for model fitting."
    )

    print(
        "[✓] Test data was not used for scaling."
    )

    print(
        "[✓] Selected model remained unchanged."
    )

    return {
        "target_column":
            target_column,

        "model":
            "Balanced Logistic Regression",

        "development_rows":
            len(development_df),

        "test_rows":
            len(test_df),

        "accuracy":
            test_metrics[
                "accuracy"
            ],

        "balanced_accuracy":
            test_metrics[
                "balanced_accuracy"
            ],

        "macro_precision":
            test_metrics[
                "macro_precision"
            ],

        "macro_recall":
            test_metrics[
                "macro_recall"
            ],

        "macro_f1":
            test_metrics[
                "macro_f1"
            ],

        "baseline_macro_f1":
            baseline_macro_f1,

        "improvement_vs_baseline":
            improvement_vs_baseline,

        "validation_macro_f1":
            validation_f1,

        "test_minus_validation":
            validation_test_difference,

        "trained_model":
            final_model,
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    results,
):
    """
    Print final Test results across all horizons.
    """

    rows = []

    for (
        target_column,
        result,
    ) in results.items():

        rows.append(
            {
                "target_column":
                    target_column,

                "model":
                    result[
                        "model"
                    ],

                "development_rows":
                    result[
                        "development_rows"
                    ],

                "test_rows":
                    result[
                        "test_rows"
                    ],

                "accuracy":
                    result[
                        "accuracy"
                    ],

                "balanced_accuracy":
                    result[
                        "balanced_accuracy"
                    ],

                "macro_precision":
                    result[
                        "macro_precision"
                    ],

                "macro_recall":
                    result[
                        "macro_recall"
                    ],

                "macro_f1":
                    result[
                        "macro_f1"
                    ],

                "baseline_macro_f1":
                    result[
                        "baseline_macro_f1"
                    ],

                "improvement_vs_baseline":
                    result[
                        "improvement_vs_baseline"
                    ],

                "validation_macro_f1":
                    result[
                        "validation_macro_f1"
                    ],

                "test_minus_validation":
                    result[
                        "test_minus_validation"
                    ],
            }
        )

    summary_df = pd.DataFrame(
        rows
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FINAL TEST EVALUATION SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FINAL MODELING PHASE COMPLETE"
    )

    print(
        "=" * 120
    )

    print(
        "\nSelected final model:"
    )

    print(
        "Balanced Logistic Regression"
    )

    print(
        "\n[✓] target_6h evaluated on Test"
    )

    print(
        "[✓] target_12h evaluated on Test"
    )

    print(
        "[✓] target_24h evaluated on Test"
    )

    print(
        "[✓] Train + Validation used for final fitting"
    )

    print(
        "[✓] Test excluded from preprocessing"
    )

    print(
        "[✓] Test excluded from training"
    )

    print(
        "[✓] Test evaluated exactly once"
    )

    print(
        "[✓] Majority baseline comparison completed"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These Test results are now final evaluation results."
    )

    print(
        "Do not tune model hyperparameters using these Test scores."
    )

    return summary_df


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run final one-time Test evaluation.
    """

    print(
        "\n"
        + "=" * 120
    )

    print(
        "RUNNING FINAL TEST EVALUATION"
    )

    print(
        "=" * 120
    )

    print(
        "\nSelected model is LOCKED:"
    )

    print(
        "Balanced Logistic Regression"
    )

    print(
        "\nTest evaluation begins now."
    )

    # --------------------------------------------------------
    # Rebuild temporal datasets from PostgreSQL pipeline
    # --------------------------------------------------------

    temporal_datasets = (
        run_analysis_pipeline()
    )

    if temporal_datasets is None:

        raise ValueError(
            "Temporal datasets were not returned "
            "by analyze_returns.main()."
        )

    final_results = {}

    # --------------------------------------------------------
    # Evaluate all horizons
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

        result = (
            evaluate_horizon(
                temporal_datasets[
                    target_column
                ],
                target_column,
            )
        )

        final_results[
            target_column
        ] = result

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    summary_df = (
        print_final_summary(
            final_results
        )
    )

    return {
        "results":
            final_results,

        "summary":
            summary_df,
    }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()