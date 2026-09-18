#!/usr/bin/env bash
# ============================================================
# AlphaPulse - Quick Start (macOS)
# macOS bootstrap + project runner
# ============================================================
# Runs on the global Python environment (no virtual env).
# Compatible with the bash 3.2 that ships with macOS.

set -euo pipefail

echo ""
echo "============================================================"
echo "        ALPHAPULSE - CRYPTO TREND PREDICTION"
echo "============================================================"
echo ""


# ------------------------------------------------------------
# Helper: stop with a readable error
# ------------------------------------------------------------

die() {
    echo ""
    echo "[ERROR] $1"
    echo ""
    exit 1
}


# ------------------------------------------------------------
# 1. Move automatically to project root
# ------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "[OK] Project directory: $PROJECT_ROOT"


# ------------------------------------------------------------
# 2. Check Python
# ------------------------------------------------------------

echo ""
echo "Checking Python installation..."

if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    die "Python was not found. Install Python 3.10+ (brew install python, or https://www.python.org) and run quick_start.sh again."
fi

if ! PYTHON_VERSION="$("$PYTHON" --version 2>&1)"; then
    die "Python was detected but could not be executed."
fi

echo "[OK] $PYTHON_VERSION at $(command -v "$PYTHON")"


# ------------------------------------------------------------
# 3. Check requirements.txt
# ------------------------------------------------------------

REQUIREMENTS="$PROJECT_ROOT/requirements.txt"

if [ ! -f "$REQUIREMENTS" ]; then
    die "requirements.txt was not found in the project root."
fi

echo "[OK] requirements.txt found."


# ------------------------------------------------------------
# 4. Check pip
# ------------------------------------------------------------

echo ""
echo "Checking pip..."

if ! "$PYTHON" -m pip --version; then
    die "pip is not available for $PYTHON. Try: $PYTHON -m ensurepip --upgrade"
fi

echo "[OK] pip is available."


# ------------------------------------------------------------
# 5. Check/install project dependencies
# ------------------------------------------------------------

echo ""
echo "Checking project dependencies..."

# Prints one line per missing/outdated requirement; exit code 1 if any.
# Uses pip's vendored 'packaging' so nothing extra needs to be installed.
check_requirements() {
    "$PYTHON" - "$REQUIREMENTS" <<'PYEOF'
import sys
from importlib import metadata
try:
    from packaging.requirements import Requirement
except ImportError:
    from pip._vendor.packaging.requirements import Requirement

problems = []
with open(sys.argv[1]) as f:
    for raw in f:
        line = raw.split(" #")[0].strip()
        # Skip blanks, comments, pip options (-r, -e, --index-url) and URL/VCS installs
        if not line or line.startswith(("#", "-")) or "://" in line:
            continue
        try:
            req = Requirement(line)
        except Exception:
            problems.append(f"  ? could not parse: {line}")
            continue
        if req.marker and not req.marker.evaluate():
            continue
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            problems.append(f"  - missing:  {req}")
            continue
        if req.specifier and not req.specifier.contains(installed, prereleases=True):
            problems.append(f"  - mismatch: {req.name} {installed} (needs {req.specifier})")

for p in problems:
    print(p)
sys.exit(1 if problems else 0)
PYEOF
}

install_requirements() {
    local log
    log="$(mktemp)"
    if "$PYTHON" -m pip install -r "$REQUIREMENTS" 2>&1 | tee "$log"; then
        rm -f "$log"
        return 0
    fi

    # Homebrew / system Python may block global installs (PEP 668)
    if grep -q "externally-managed-environment" "$log"; then
        rm -f "$log"
        echo ""
        echo "[WARN] This Python is marked as externally managed (PEP 668)."
        echo "       Installing globally requires --break-system-packages,"
        echo "       which can conflict with Homebrew-managed packages."
        read -r -p "Proceed anyway? [y/N] " force
        case "$force" in
            [yY]|[yY][eE][sS])
                "$PYTHON" -m pip install --break-system-packages -r "$REQUIREMENTS"
                return $?
                ;;
            *)
                return 1
                ;;
        esac
    fi

    rm -f "$log"
    return 1
}

if check_output="$(check_requirements)"; then
    echo "[OK] Project dependencies are installed."
else
    echo "[WARN] Some dependencies are missing or outdated:"
    echo "$check_output"
    echo ""
    read -r -p "Install them now into the global environment? [Y/n] " answer
    case "$answer" in
        [nN]|[nN][oO])
            die "Cannot continue without required dependencies."
            ;;
        *)
            if ! install_requirements; then
                die "Dependency installation failed. Check requirements.txt and your internet connection."
            fi
            if ! check_requirements >/dev/null; then
                die "Dependencies still not satisfied after install."
            fi
            echo "[OK] Project dependencies are installed."
            ;;
    esac
fi


# ------------------------------------------------------------
# 6. Check .env
# ------------------------------------------------------------

ENV_FILE="$PROJECT_ROOT/.env"

echo ""
echo "Checking environment configuration..."

if [ ! -f "$ENV_FILE" ]; then
    echo ""
    echo "[ERROR] .env file was not found."
    echo ""
    echo "Create:"
    echo "  $ENV_FILE"
    echo ""
    echo "Required configuration includes:"
    echo "  DATABASE_URL"
    echo "  COINGECKO_API_KEY"
    echo ""
    echo "Optional LLM functionality requires:"
    echo "  GROQ_API_KEY"
    echo ""
    exit 1
fi

echo "[OK] .env file found."


# ------------------------------------------------------------
# 7. Validate required environment variables through Python
# ------------------------------------------------------------

echo ""
echo "Validating required environment variables..."

validate_env() {
    "$PYTHON" - <<'PYEOF'
import os
import sys
from dotenv import load_dotenv

# The script already changed the working directory to the
# project root, so pass the .env path explicitly. This avoids
# python-dotenv inspecting the call stack when code runs via stdin.
load_dotenv(dotenv_path=".env")

required = [
    "DATABASE_URL",
    "COINGECKO_API_KEY",
]

missing = [name for name in required if not os.getenv(name)]

if missing:
    print("[ERROR] Missing required environment variable(s):")
    for name in missing:
        print(f"  - {name}")
    sys.exit(1)

print("[OK] DATABASE_URL is configured.")
print("[OK] COINGECKO_API_KEY is configured.")

if os.getenv("GROQ_API_KEY"):
    print("[OK] GROQ_API_KEY is configured.")
else:
    print("[INFO] GROQ_API_KEY is not configured. AI explanations may be unavailable.")
PYEOF
}

if ! validate_env; then
    die "Environment validation failed."
fi


# ------------------------------------------------------------
# 8. Test database connectivity
# ------------------------------------------------------------

echo ""
echo "Testing PostgreSQL connection..."

check_database() {
    "$PYTHON" - <<'PYEOF'
import sys

try:
    from sqlalchemy import text
    from src.database.connection import engine

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    print("[OK] PostgreSQL connection successful.")

except Exception as exc:
    print("[ERROR] PostgreSQL connection failed.")
    print(str(exc))
    sys.exit(1)
PYEOF
}

if ! check_database; then
    die "Could not connect to PostgreSQL using DATABASE_URL."
fi


# ------------------------------------------------------------
# 9. Check production model artifacts
# ------------------------------------------------------------

echo ""
echo "Checking production model artifacts..."

# Plain string instead of an array: empty arrays trip 'set -u' in bash 3.2
MISSING_MODELS=""

for model_path in \
    "artifacts/models/model_6h.joblib" \
    "artifacts/models/model_12h.joblib" \
    "artifacts/models/model_24h.joblib"
do
    if [ ! -f "$PROJECT_ROOT/$model_path" ]; then
        MISSING_MODELS="${MISSING_MODELS}  - ${model_path}
"
    fi
done

if [ -z "$MISSING_MODELS" ]; then
    echo "[OK] Production model artifacts found."
else
    echo "[WARNING] Some production model artifacts are missing:"
    printf "%s" "$MISSING_MODELS"
    echo ""
    echo "[INFO] You can generate production models using menu option 8."
fi


# ------------------------------------------------------------
# 10. Environment ready
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "ALPHAPULSE ENVIRONMENT READY"
echo "============================================================"
echo ""


# ------------------------------------------------------------
# 11. Show menu
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
# 12. Run selected stage
# ------------------------------------------------------------

case "$choice" in
    1) message="Running analysis and temporal dataset pipeline..."; module="src.analysis.analyze_returns" ;;
    2) message="Running majority baseline...";                      module="src.modeling.run_baselines" ;;
    3) message="Training Logistic Regression...";                   module="src.modeling.train_logistic_regression" ;;
    4) message="Training Random Forest...";                         module="src.modeling.train_random_forest" ;;
    5) message="Training XGBoost...";                               module="src.modeling.train_xgboost" ;;
    6) message="Comparing models...";                               module="src.modeling.compare_models" ;;
    7) message="Running final test evaluation...";                  module="src.modeling.evaluate_final_test" ;;
    8) message="Training production models...";                     module="src.modeling.train_production_models" ;;
    9)
        echo ""
        echo "Exiting."
        exit 0
        ;;
    *)
        die "Invalid option."
        ;;
esac

echo ""
echo "$message"


# ------------------------------------------------------------
# 13. Validate selected command
# ------------------------------------------------------------

set +e
"$PYTHON" -m "$module"
exit_code=$?
set -e

if [ "$exit_code" -ne 0 ]; then
    echo ""
    echo "[ERROR] Program failed with exit code $exit_code."
    exit "$exit_code"
fi

echo ""
echo "============================================================"
echo "[DONE] Command completed successfully."
echo "============================================================"