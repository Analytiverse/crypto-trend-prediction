"""
Model Comparison
================

Compare all completed models across:

- target_6h
- target_12h
- target_24h

Models:
- Majority Baseline
- Logistic Regression
- Random Forest
- XGBoost

Primary selection metric:
- Macro F1

Secondary metrics:
- Balanced Accuracy
- Accuracy

Important:
- This file does NOT retrain models.
- This file does NOT touch the Test set.
- It summarizes already completed Validation results.
"""


import pandas as pd


# ============================================================
# RESULTS FROM COMPLETED VALIDATION RUNS
# ============================================================

MODEL_RESULTS = {
    "target_6h": {
        "Majority Baseline": {
            "accuracy": 0.430588,
            "balanced_accuracy": 0.333333,
            "macro_f1": 0.200658,
        },

        "Balanced Logistic Regression": {
            "accuracy": 0.486471,
            "balanced_accuracy": 0.453279,
            "macro_f1": 0.449475,
        },

        "Balanced Random Forest": {
            "accuracy": 0.441176,
            "balanced_accuracy": 0.429520,
            "macro_f1": 0.425116,
        },

        "Balanced XGBoost": {
            "accuracy": 0.438235,
            "balanced_accuracy": 0.432436,
            "macro_f1": 0.428308,
        },
    },

    "target_12h": {
        "Majority Baseline": {
            "accuracy": 0.541916,
            "balanced_accuracy": 0.333333,
            "macro_f1": 0.234304,
        },

        "Balanced Logistic Regression": {
            "accuracy": 0.580838,
            "balanced_accuracy": 0.520416,
            "macro_f1": 0.513312,
        },

        "Standard Random Forest": {
            "accuracy": 0.521557,
            "balanced_accuracy": 0.416924,
            "macro_f1": 0.411406,
        },

        "Balanced XGBoost": {
            "accuracy": 0.447904,
            "balanced_accuracy": 0.428530,
            "macro_f1": 0.410791,
        },
    },

    "target_24h": {
        "Majority Baseline": {
            "accuracy": 0.554375,
            "balanced_accuracy": 0.333333,
            "macro_f1": 0.237770,
        },

        "Balanced Logistic Regression": {
            "accuracy": 0.581875,
            "balanced_accuracy": 0.524343,
            "macro_f1": 0.512767,
        },

        "Standard Random Forest": {
            "accuracy": 0.496250,
            "balanced_accuracy": 0.388536,
            "macro_f1": 0.382788,
        },

        "Balanced XGBoost": {
            "accuracy": 0.455625,
            "balanced_accuracy": 0.416538,
            "macro_f1": 0.400190,
        },
    },
}


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

def build_comparison_dataframe():
    """
    Convert nested results dictionary
    into one clean DataFrame.
    """

    rows = []

    for target_column, models in MODEL_RESULTS.items():

        for model_name, metrics in models.items():

            rows.append(
                {
                    "target_column": target_column,
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics[
                        "balanced_accuracy"
                    ],
                    "macro_f1": metrics["macro_f1"],
                }
            )

    comparison_df = pd.DataFrame(
        rows
    )

    return comparison_df


# ============================================================
# RANK MODELS WITHIN EACH HORIZON
# ============================================================

def rank_models(
    comparison_df,
):
    """
    Rank models using Macro F1.

    Rank 1 = best model.
    """

    ranked_df = (
        comparison_df
        .copy()
    )

    ranked_df[
        "macro_f1_rank"
    ] = (
        ranked_df
        .groupby(
            "target_column"
        )[
            "macro_f1"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    ranked_df = (
        ranked_df
        .sort_values(
            by=[
                "target_column",
                "macro_f1_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return ranked_df


# ============================================================
# SELECT BEST MODEL PER HORIZON
# ============================================================

def select_best_models(
    ranked_df,
):
    """
    Select rank-1 model for each horizon.
    """

    best_models = (
        ranked_df[
            ranked_df[
                "macro_f1_rank"
            ]
            == 1
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return best_models


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(
    ranked_df,
):
    """
    Print all models grouped by horizon.
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 100
    )

    for target_column in [
        "target_6h",
        "target_12h",
        "target_24h",
    ]:

        print(
            "\n"
            + "-" * 100
        )

        print(
            target_column
        )

        print(
            "-" * 100
        )

        target_results = (
            ranked_df[
                ranked_df[
                    "target_column"
                ]
                == target_column
            ]
            .copy()
        )

        print(
            target_results[
                [
                    "macro_f1_rank",
                    "model",
                    "macro_f1",
                    "balanced_accuracy",
                    "accuracy",
                ]
            ]
            .to_string(
                index=False
            )
        )


# ============================================================
# PRINT FINAL MODEL SELECTION
# ============================================================

def print_final_selection(
    best_models,
):
    """
    Print selected final model
    for each horizon.
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL MODEL SELECTION"
    )

    print(
        "=" * 100
    )

    for _, row in best_models.iterrows():

        print(
            f"\n{row['target_column']}"
        )

        print(
            "-" * 70
        )

        print(
            f"Selected model: "
            f"{row['model']}"
        )

        print(
            f"Validation Macro F1: "
            f"{row['macro_f1']:.6f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{row['balanced_accuracy']:.6f}"
        )

        print(
            f"Accuracy: "
            f"{row['accuracy']:.6f}"
        )


# ============================================================
# IMPROVEMENT OVER BASELINE
# ============================================================

def print_baseline_improvement(
    comparison_df,
    best_models,
):
    """
    Compare selected model against majority baseline.
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        "IMPROVEMENT OVER MAJORITY BASELINE"
    )

    print(
        "=" * 100
    )

    for _, best_row in best_models.iterrows():

        target_column = (
            best_row[
                "target_column"
            ]
        )

        baseline_row = (
            comparison_df[
                (
                    comparison_df[
                        "target_column"
                    ]
                    == target_column
                )
                &
                (
                    comparison_df[
                        "model"
                    ]
                    == "Majority Baseline"
                )
            ]
            .iloc[0]
        )

        difference = (
            best_row[
                "macro_f1"
            ]
            -
            baseline_row[
                "macro_f1"
            ]
        )

        relative_improvement = (
            difference
            /
            baseline_row[
                "macro_f1"
            ]
            *
            100
        )

        print(
            f"\n{target_column}"
        )

        print(
            "-" * 70
        )

        print(
            f"Baseline Macro F1: "
            f"{baseline_row['macro_f1']:.6f}"
        )

        print(
            f"Selected-model Macro F1: "
            f"{best_row['macro_f1']:.6f}"
        )

        print(
            f"Absolute improvement: "
            f"{difference:+.6f}"
        )

        print(
            f"Relative improvement: "
            f"{relative_improvement:+.2f}%"
        )


# ============================================================
# CONSISTENCY CHECK
# ============================================================

def print_consistency_check(
    best_models,
):
    """
    Check whether same model wins
    all prediction horizons.
    """

    selected_models = (
        best_models[
            "model"
        ]
        .unique()
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "MODEL CONSISTENCY CHECK"
    )

    print(
        "=" * 100
    )

    if len(
        selected_models
    ) == 1:

        print(
            "\n[✓] The same model wins "
            "all prediction horizons."
        )

        print(
            f"Winning model: "
            f"{selected_models[0]}"
        )

    else:

        print(
            "\n[INFO] Different horizons "
            "select different models."
        )

        for _, row in best_models.iterrows():

            print(
                f"{row['target_column']}: "
                f"{row['model']}"
            )


# ============================================================
# TEST STATUS
# ============================================================

def print_test_status():
    """
    Confirm Test remains untouched.
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TEST-SET STATUS"
    )

    print(
        "=" * 100
    )

    print(
        "\nTest predictions generated: NO"
    )

    print(
        "Test evaluation performed: NO"
    )

    print(
        "[✓] Test remains untouched."
    )

    print(
        "[✓] Final model selection "
        "used Validation only."
    )

    print(
        "\nThe next phase will perform "
        "the one-time final Test evaluation."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run formal model comparison.
    """

    comparison_df = (
        build_comparison_dataframe()
    )

    ranked_df = (
        rank_models(
            comparison_df
        )
    )

    best_models = (
        select_best_models(
            ranked_df
        )
    )

    print_comparison(
        ranked_df
    )

    print_final_selection(
        best_models
    )

    print_baseline_improvement(
        comparison_df,
        best_models,
    )

    print_consistency_check(
        best_models
    )

    print_test_status()

    print(
        "\n"
        + "=" * 100
    )

    print(
        "MODEL COMPARISON PHASE COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        "\n[✓] Majority baseline compared"
    )

    print(
        "[✓] Logistic Regression compared"
    )

    print(
        "[✓] Random Forest compared"
    )

    print(
        "[✓] XGBoost compared"
    )

    print(
        "[✓] Macro F1 used as primary metric"
    )

    print(
        "[✓] Final model selected per horizon"
    )

    print(
        "[✓] Test remains untouched"
    )

    return {
        "comparison":
            ranked_df,

        "selected_models":
            best_models,
    }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()