#!/bin/bash

# ============================================================
# AlphaPulse - Quick Start
# macOS launcher
# ============================================================

set -e

echo ""
echo "============================================================"
echo "        ALPHAPULSE - CRYPTO TREND PREDICTION"
echo "============================================================"
echo ""

# ------------------------------------------------------------
# Helper: stop with a readable error
# ------------------------------------------------------------

stop_with_error() {
    echo ""
    echo "[ERROR] $1"
    echo ""
    exit 1
}


# ------------------------------------------------------------
# 1. Move automatically to project root
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_ROOT="$SCRIPT_DIR"

echo "[OK] Project directory: $PROJECT_ROOT"


# ------------------------------------------------------------
# 2. Check Python
# ------------------------------------------------------------

echo ""
echo "Checking Python installation..."

if command -v python3 >/dev/null 2>&1; then
    PYTHON_COMMAND="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_COMMAND="python"
else
    stop_with_error "Python was not found. Install Python 3.10+ and run quick_start.sh again."
fi

PYTHON_VERSION="$($PYTHON_COMMAND --version 2>&1)"

echo "[OK] $PYTHON_VERSION"


# ------------------------------------------------------------
# 3. Create virtual environment when missing
# ------------------------------------------------------------

VENV_DIRECTORY="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_DIRECTORY/bin/python"
VENV_ACTIVATE="$VENV_DIRECTORY/bin/activate"

if [ ! -f "$VENV_PYTHON" ]; then

    echo ""
    echo "[INFO] Virtual environment was not found."
    echo "[INFO] Creating .venv..."

    "$PYTHON_COMMAND" -m venv "$VENV_DIRECTORY" || \
        stop_with_error "Failed to create the virtual environment."

    if [ ! -f "$VENV_PYTHON" ]; then
        stop_with_error "Virtual environment creation completed but Python was not found inside .venv."
    fi

    echo "[OK] Virtual environment created."
else
    echo "[OK] Virtual environment already exists."
fi


# ------------------------------------------------------------
# 4. Activate virtual environment
# ------------------------------------------------------------

if [ ! -f "$VENV_ACTIVATE" ]; then
    stop_with_error "Virtual environment activation script was not found: $VENV_ACTIVATE"
fi

source "$VENV_ACTIVATE"

echo "[OK] Virtual environment ready."


# ------------------------------------------------------------
# 5. Check requirements.txt
# ------------------------------------------------------------

REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    stop_with_error "requirements.txt was not found in the project root."
fi

echo "[OK] requirements.txt found."


# ------------------------------------------------------------
# 6. Check pip
# ------------------------------------------------------------

echo ""
echo "Checking pip..."

"$VENV_PYTHON" -m pip --version || \
    stop_with_error "pip is not available inside the virtual environment."

echo "[OK] pip is available."


# ------------------------------------------------------------
# 7. Install project dependencies
# ------------------------------------------------------------

echo ""
echo "Checking project dependencies..."

"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE" || \
    stop_with_error "Dependency installation failed. Check requirements.txt and your internet connection."

echo "[OK] Project dependencies are installed."


# ------------------------------------------------------------
# 8. Run shared AlphaPulse bootstrap
# ------------------------------------------------------------

echo ""
echo "Running shared AlphaPulse bootstrap..."

SHARED_BOOTSTRAP="$PROJECT_ROOT/src/bootstrap.py"

if [ ! -f "$SHARED_BOOTSTRAP" ]; then
    stop_with_error "Shared bootstrap module was not found: $SHARED_BOOTSTRAP"
fi

"$VENV_PYTHON" -m src.bootstrap || \
    stop_with_error "Shared AlphaPulse bootstrap failed."


# ------------------------------------------------------------
# 9. Environment ready
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "ALPHAPULSE MACOS ENVIRONMENT READY"
echo "============================================================"
echo ""


# ------------------------------------------------------------
# 10. Show menu
# ------------------------------------------------------------

echo "What do you want to run?"
echo ""
echo "1 - Full analysis + dataset preparation"
echo "2 - Majority baseline"
echo "3 - Logistic Regression"
echo "4 - Random Forest"
echo "5 - XGBoost"
echo "6 - Compare models"
echo "7 - Final test evaluation"
echo "8 - Train production models"
echo "9 - Exit"
echo ""

read -r -p "Enter option: " choice


# ------------------------------------------------------------
# 11. Run selected stage
# ------------------------------------------------------------

case "$choice" in

    1)
        echo ""
        echo "Running analysis and temporal dataset pipeline..."
        "$VENV_PYTHON" -m src.analysis.analyze_returns
        ;;

    2)
        echo ""
        echo "Running majority baseline..."
        "$VENV_PYTHON" -m src.modeling.run_baselines
        ;;

    3)
        echo ""
        echo "Training Logistic Regression..."
        "$VENV_PYTHON" -m src.modeling.train_logistic_regression
        ;;

    4)
        echo ""
        echo "Training Random Forest..."
        "$VENV_PYTHON" -m src.modeling.train_random_forest
        ;;

    5)
        echo ""
        echo "Training XGBoost..."
        "$VENV_PYTHON" -m src.modeling.train_xgboost
        ;;

    6)
        echo ""
        echo "Comparing models..."
        "$VENV_PYTHON" -m src.modeling.compare_models
        ;;

    7)
        echo ""
        echo "Running final test evaluation..."
        "$VENV_PYTHON" -m src.modeling.evaluate_final_test
        ;;

    8)
        echo ""
        echo "Training production models..."
        "$VENV_PYTHON" -m src.modeling.train_production_models
        ;;

    9)
        echo ""
        echo "Exiting."
        exit 0
        ;;

    *)
        stop_with_error "Invalid option."
        ;;
esac


# ------------------------------------------------------------
# 12. Complete
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "[DONE] Command completed successfully."
echo "============================================================"