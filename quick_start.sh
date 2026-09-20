#!/usr/bin/env bash

# ============================================================
# AlphaPulse - Quick Start
# macOS / Linux launcher
# ============================================================

set -e


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
# Helper: detect placeholder values
# ------------------------------------------------------------

is_placeholder() {

    local value
    value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"

    case "$value" in
        ""|\
        "your_api_key"|\
        "your-api-key"|\
        "your_key"|\
        "your-key"|\
        "changeme"|\
        "change_me"|\
        "replace_me"|\
        "replace-me"|\
        "placeholder"|\
        "<your_api_key>"|\
        "<your_database_url>"|\
        "<your_cron_secret>"|\
        "<your_groq_api_key>")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}


# ------------------------------------------------------------
# Helper: read variable from .env
# ------------------------------------------------------------

get_env_file_value() {

    local variable_name="$1"

    "$VENV_PYTHON" - "$ENV_FILE" "$variable_name" <<'PY'
import sys
from pathlib import Path

env_file = Path(sys.argv[1])
name = sys.argv[2]

if not env_file.exists():
    raise SystemExit(0)

for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():

    line = raw_line.strip()

    if not line or line.startswith("#") or "=" not in line:
        continue

    key, value = line.split("=", 1)

    if key.strip() != name:
        continue

    value = value.strip()

    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in ("'", '"')
    ):
        value = value[1:-1]

    print(value, end="")
    break
PY
}


# ------------------------------------------------------------
# Helper: safely update/add variable in .env
# ------------------------------------------------------------

set_env_file_value() {

    local variable_name="$1"
    local variable_value="$2"

    "$VENV_PYTHON" - \
        "$ENV_FILE" \
        "$variable_name" \
        "$variable_value" <<'PY'

import sys
from pathlib import Path

env_file = Path(sys.argv[1])
name = sys.argv[2]
value = sys.argv[3]

if env_file.exists():
    lines = env_file.read_text(
        encoding="utf-8-sig"
    ).splitlines()
else:
    lines = []

updated = []
found = False

for line in lines:

    stripped = line.strip()

    if (
        stripped
        and not stripped.startswith("#")
        and "=" in stripped
        and stripped.split("=", 1)[0].strip() == name
    ):
        updated.append(f"{name}={value}")
        found = True
    else:
        updated.append(line)


if not found:

    if updated and updated[-1].strip():
        updated.append("")

    updated.append(f"{name}={value}")


env_file.write_text(
    "\n".join(updated) + "\n",
    encoding="utf-8"
)
PY
}


# ------------------------------------------------------------
# Helper: securely read missing credential
# ------------------------------------------------------------

read_secret() {

    local variable_name="$1"
    local entered_value

    printf "Enter value for %s: " "$variable_name" >&2

    IFS= read -r -s entered_value

    echo "" >&2

    printf '%s' "$entered_value"
}


# ------------------------------------------------------------
# 1. Move automatically to project root
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_ROOT="$SCRIPT_DIR"


echo ""
echo "============================================================"
echo "        ALPHAPULSE - CRYPTO TREND PREDICTION"
echo "============================================================"
echo ""

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

    stop_with_error \
        "Python was not found. Install Python 3.10+ and run quick_start.sh again."

fi


PYTHON_VERSION="$("$PYTHON_COMMAND" --version 2>&1)"

echo "[OK] $PYTHON_VERSION"


# ------------------------------------------------------------
# 3. Validate Python >= 3.10
# ------------------------------------------------------------

echo "Checking Python version compatibility..."


VERSION_CHECK="$(
    "$PYTHON_COMMAND" -c \
    'import sys; print(1 if sys.version_info >= (3,10) else 0)'
)"


if [ "$VERSION_CHECK" != "1" ]; then

    stop_with_error \
        "AlphaPulse requires Python 3.10 or newer. Detected: $PYTHON_VERSION"

fi


echo "[OK] Python version is compatible."


# ------------------------------------------------------------
# 4. Check important project files
# ------------------------------------------------------------

echo ""
echo "Checking project files..."


REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
SHARED_BOOTSTRAP="$PROJECT_ROOT/src/bootstrap.py"


if [ ! -f "$REQUIREMENTS_FILE" ]; then

    stop_with_error \
        "requirements.txt was not found in the project root."

fi


if [ ! -f "$SHARED_BOOTSTRAP" ]; then

    stop_with_error \
        "Shared bootstrap module was not found: $SHARED_BOOTSTRAP"

fi


echo "[OK] requirements.txt found."
echo "[OK] src/bootstrap.py found."


# ------------------------------------------------------------
# 5. Create virtual environment when missing
# ------------------------------------------------------------

VENV_DIRECTORY="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_DIRECTORY/bin/python"
VENV_ACTIVATE="$VENV_DIRECTORY/bin/activate"


if [ ! -f "$VENV_PYTHON" ]; then

    echo ""
    echo "[INFO] Virtual environment was not found."
    echo "[INFO] Creating .venv..."


    "$PYTHON_COMMAND" -m venv "$VENV_DIRECTORY" || \
        stop_with_error \
        "Failed to create the virtual environment."


    if [ ! -f "$VENV_PYTHON" ]; then

        stop_with_error \
            "Virtual environment creation completed but Python was not found inside .venv."

    fi


    echo "[OK] Virtual environment created."

else

    echo "[OK] Virtual environment already exists."

fi


# ------------------------------------------------------------
# 6. Activate virtual environment
# ------------------------------------------------------------

if [ ! -f "$VENV_ACTIVATE" ]; then

    stop_with_error \
        "Virtual environment activation script was not found: $VENV_ACTIVATE"

fi


# shellcheck disable=SC1090
source "$VENV_ACTIVATE"


echo "[OK] Virtual environment ready."


# ------------------------------------------------------------
# 7. Check pip
# ------------------------------------------------------------

echo ""
echo "Checking pip..."


"$VENV_PYTHON" -m pip --version || \
    stop_with_error \
    "pip is not available inside the virtual environment."


echo "[OK] pip is available."


# ------------------------------------------------------------
# 8. Upgrade pip
# ------------------------------------------------------------

echo ""
echo "Checking/upgrading pip..."


if "$VENV_PYTHON" -m pip install --upgrade pip; then

    echo "[OK] pip is up to date."

else

    echo "[WARNING] pip upgrade failed."
    echo "[INFO] Continuing with the installed pip version."

fi


# ------------------------------------------------------------
# 9. Install/sync project dependencies
# ------------------------------------------------------------

echo ""
echo "Installing/syncing project dependencies..."


"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE" || \
    stop_with_error \
    "Dependency installation failed. Check requirements.txt and your internet connection."


echo "[OK] Project dependencies are installed."


# ------------------------------------------------------------
# 10. Prepare .env
# ------------------------------------------------------------

echo ""
echo "Checking environment configuration..."


ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE_FILE="$PROJECT_ROOT/.env.example"


if [ ! -f "$ENV_FILE" ]; then

    echo "[INFO] .env file was not found."


    if [ -f "$ENV_EXAMPLE_FILE" ]; then

        echo "[INFO] Creating .env from .env.example..."

        cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"

        echo "[OK] .env created from .env.example."

    else

        echo "[INFO] .env.example was not found."
        echo "[INFO] Creating empty .env..."

        touch "$ENV_FILE"

        echo "[OK] Empty .env created."

    fi

else

    echo "[OK] .env file found."

fi


# ------------------------------------------------------------
# 11. Check .gitignore protection
# ------------------------------------------------------------

GITIGNORE_FILE="$PROJECT_ROOT/.gitignore"


if [ -f "$GITIGNORE_FILE" ]; then

    if grep -Eq '^[[:space:]]*/?\.env[[:space:]]*$' \
        "$GITIGNORE_FILE"; then

        echo "[OK] .env is protected by .gitignore."

    else

        echo "[WARNING] .env is not listed in .gitignore."
        echo "[WARNING] Add '.env' before committing."

    fi

else

    echo "[WARNING] .gitignore was not found."
    echo "[WARNING] Make sure .env is never committed."

fi


# ------------------------------------------------------------
# 12. Check / assign required environment variables
# ------------------------------------------------------------

echo ""
echo "Checking required environment variables..."


REQUIRED_ENV_VARS=(
    "COINGECKO_API_KEY"
    "DATABASE_URL"
    "CRON_SECRET"
    "GROQ_API_KEY"
)


for VARIABLE_NAME in "${REQUIRED_ENV_VARS[@]}"; do

    # --------------------------------------------------------
    # Priority 1: current environment
    # --------------------------------------------------------

    CURRENT_VALUE="${!VARIABLE_NAME:-}"


    if [ -n "$CURRENT_VALUE" ] && \
       ! is_placeholder "$CURRENT_VALUE"; then

        echo "[OK] $VARIABLE_NAME already assigned in environment."

        continue

    fi


    # --------------------------------------------------------
    # Priority 2: .env
    # --------------------------------------------------------

    DOTENV_VALUE="$(get_env_file_value "$VARIABLE_NAME")"


    if [ -n "$DOTENV_VALUE" ] && \
       ! is_placeholder "$DOTENV_VALUE"; then

        export "$VARIABLE_NAME=$DOTENV_VALUE"

        echo "[OK] $VARIABLE_NAME loaded from .env and assigned."

        continue

    fi


    # --------------------------------------------------------
    # Priority 3: ask once, save and assign
    # --------------------------------------------------------

    echo ""
    echo "[INFO] $VARIABLE_NAME is not configured."
    echo "[INFO] Enter it once. It will be saved in .env."


    ENTERED_VALUE="$(read_secret "$VARIABLE_NAME")"


    if [ -z "$ENTERED_VALUE" ] || \
       is_placeholder "$ENTERED_VALUE"; then

        stop_with_error \
            "$VARIABLE_NAME must contain a valid value."

    fi


    # Basic DATABASE_URL validation
    if [ "$VARIABLE_NAME" = "DATABASE_URL" ]; then

        case "$ENTERED_VALUE" in

            postgresql://*|postgres://*)
                ;;

            *)
                stop_with_error \
                    "DATABASE_URL must start with postgresql:// or postgres://"
                ;;

        esac

    fi


    set_env_file_value \
        "$VARIABLE_NAME" \
        "$ENTERED_VALUE"


    export "$VARIABLE_NAME=$ENTERED_VALUE"


    echo "[OK] $VARIABLE_NAME saved to .env and assigned."

done


# ------------------------------------------------------------
# 13. Final environment verification
# ------------------------------------------------------------

echo ""
echo "Verifying environment variables..."


for VARIABLE_NAME in "${REQUIRED_ENV_VARS[@]}"; do

    FINAL_VALUE="${!VARIABLE_NAME:-}"


    if [ -z "$FINAL_VALUE" ] || \
       is_placeholder "$FINAL_VALUE"; then

        stop_with_error \
            "$VARIABLE_NAME is still missing after environment setup."

    fi


    echo "[OK] $VARIABLE_NAME ready."

done


echo "[OK] All required environment variables are assigned."


# ------------------------------------------------------------
# 14. Run shared AlphaPulse bootstrap
# ------------------------------------------------------------

echo ""
echo "Running shared AlphaPulse bootstrap..."


"$VENV_PYTHON" -m src.bootstrap || \
    stop_with_error \
    "Shared AlphaPulse bootstrap failed."


# ------------------------------------------------------------
# 15. Environment ready
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "ALPHAPULSE MACOS/LINUX ENVIRONMENT READY"
echo "============================================================"
echo ""


# ------------------------------------------------------------
# 16. Show menu
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
# 17. Run selected stage
# ------------------------------------------------------------

case "$choice" in

    1)
        echo ""
        echo "Running analysis and temporal dataset pipeline..."

        "$VENV_PYTHON" \
            -m src.analysis.analyze_returns
        ;;


    2)
        echo ""
        echo "Running majority baseline..."

        "$VENV_PYTHON" \
            -m src.ml.evaluation.run_baselines
        ;;


    3)
        echo ""
        echo "Training Logistic Regression..."

        "$VENV_PYTHON" \
            -m src.ml.training.train_logistic_regression
        ;;


    4)
        echo ""
        echo "Training Random Forest..."

        "$VENV_PYTHON" \
            -m src.ml.training.train_random_forest
        ;;


    5)
        echo ""
        echo "Training XGBoost..."

        "$VENV_PYTHON" \
            -m src.ml.training.train_xgboost
        ;;


    6)
        echo ""
        echo "Comparing models..."

        "$VENV_PYTHON" \
            -m src.ml.evaluation.compare_models
        ;;


    7)
        echo ""
        echo "Running final test evaluation..."

        "$VENV_PYTHON" \
            -m src.ml.evaluation.evaluate_final_test
        ;;


    8)
        echo ""
        echo "Training production models..."

        "$VENV_PYTHON" \
            -m src.ml.training.train_production_models
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
# 18. Complete
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "[DONE] Command completed successfully."
echo "============================================================"