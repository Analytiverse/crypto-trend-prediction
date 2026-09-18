# ============================================================
# AlphaPulse - Quick Start
# Windows bootstrap + project runner
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================"
Write-Host "        ALPHAPULSE - CRYPTO TREND PREDICTION"
Write-Host "============================================================"
Write-Host ""


# ------------------------------------------------------------
# Helper: stop with a readable error
# ------------------------------------------------------------

function Stop-WithError {
    param(
        [string]$Message
    )

    Write-Host ""
    Write-Host "[ERROR] $Message"
    Write-Host ""
    exit 1
}


# ------------------------------------------------------------
# 1. Move automatically to project root
# ------------------------------------------------------------

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "[OK] Project directory: $ProjectRoot"


# ------------------------------------------------------------
# 2. Check Python
# ------------------------------------------------------------

Write-Host ""
Write-Host "Checking Python installation..."

$PythonCommand = $null

if (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCommand = "python"
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCommand = "py"
}
else {
    Stop-WithError "Python was not found. Install Python 3.10+ and run quick_start.ps1 again."
}

try {
    if ($PythonCommand -eq "py") {
        $PythonVersion = & py --version 2>&1
    }
    else {
        $PythonVersion = & python --version 2>&1
    }
}
catch {
    Stop-WithError "Python was detected but could not be executed."
}

Write-Host "[OK] $PythonVersion"


# ------------------------------------------------------------
# 3. Create virtual environment when missing
# ------------------------------------------------------------

$VenvDirectory = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDirectory "Scripts\python.exe"
$VenvActivate = Join-Path $VenvDirectory "Scripts\Activate.ps1"

if (-not (Test-Path $VenvPython)) {

    Write-Host ""
    Write-Host "[INFO] Virtual environment was not found."
    Write-Host "[INFO] Creating .venv..."

    if ($PythonCommand -eq "py") {
        & py -m venv .venv
    }
    else {
        & python -m venv .venv
    }

    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "Failed to create the virtual environment."
    }

    if (-not (Test-Path $VenvPython)) {
        Stop-WithError "Virtual environment creation completed but Python was not found inside .venv."
    }

    Write-Host "[OK] Virtual environment created."
}
else {
    Write-Host "[OK] Virtual environment already exists."
}


# ------------------------------------------------------------
# 4. Activate virtual environment
# ------------------------------------------------------------

if (-not (Test-Path $VenvActivate)) {
    Stop-WithError "Virtual environment activation script was not found: $VenvActivate"
}

try {
    & $VenvActivate
}
catch {
    Write-Host ""
    Write-Host "[WARNING] PowerShell blocked virtual environment activation."
    Write-Host "[INFO] AlphaPulse can still continue using .venv Python directly."
}

Write-Host "[OK] Virtual environment ready."


# ------------------------------------------------------------
# 5. Check requirements.txt
# ------------------------------------------------------------

$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"

if (-not (Test-Path $RequirementsFile)) {
    Stop-WithError "requirements.txt was not found in the project root."
}

Write-Host "[OK] requirements.txt found."


# ------------------------------------------------------------
# 6. Check pip
# ------------------------------------------------------------

Write-Host ""
Write-Host "Checking pip..."

& $VenvPython -m pip --version

if ($LASTEXITCODE -ne 0) {
    Stop-WithError "pip is not available inside the virtual environment."
}

Write-Host "[OK] pip is available."


# ------------------------------------------------------------
# 7. Check/install project dependencies
# ------------------------------------------------------------

Write-Host ""
Write-Host "Checking project dependencies..."

# Running pip install -r requirements.txt repeatedly is safe.
# Packages that already satisfy requirements will be kept.
# Missing packages will be installed automatically.

& $VenvPython -m pip install -r $RequirementsFile

if ($LASTEXITCODE -ne 0) {
    Stop-WithError "Dependency installation failed. Check requirements.txt and your internet connection."
}

Write-Host "[OK] Project dependencies are installed."


# ------------------------------------------------------------
# 8. Check .env
# ------------------------------------------------------------

$EnvFile = Join-Path $ProjectRoot ".env"

Write-Host ""
Write-Host "Checking environment configuration..."

if (-not (Test-Path $EnvFile)) {

    Write-Host ""
    Write-Host "[ERROR] .env file was not found."
    Write-Host ""
    Write-Host "Create:"
    Write-Host "  $EnvFile"
    Write-Host ""
    Write-Host "Required configuration includes:"
    Write-Host "  DATABASE_URL"
    Write-Host "  COINGECKO_API_KEY"
    Write-Host ""
    Write-Host "Optional LLM functionality requires:"
    Write-Host "  GROQ_API_KEY"
    Write-Host ""

    exit 1
}

Write-Host "[OK] .env file found."


# ------------------------------------------------------------
# 9. Validate required environment variables through Python
# ------------------------------------------------------------

Write-Host ""
Write-Host "Validating required environment variables..."

$EnvCheck = @'
import os
import sys
from dotenv import load_dotenv

# quick_start.ps1 already changes the working directory
# to the project root, so use the .env path explicitly.
# This avoids python-dotenv trying to inspect the call stack
# when Python code is executed through stdin.
load_dotenv(dotenv_path=".env")

required = [
    "DATABASE_URL",
    "COINGECKO_API_KEY",
]

missing = [
    name
    for name in required
    if not os.getenv(name)
]

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
'@

$EnvCheck | & $VenvPython -

if ($LASTEXITCODE -ne 0) {
    Stop-WithError "Environment validation failed."
}


# ------------------------------------------------------------
# 10. Test database connectivity
# ------------------------------------------------------------

Write-Host ""
Write-Host "Testing PostgreSQL connection..."

$DatabaseCheck = @'
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
'@

$DatabaseCheck | & $VenvPython -

if ($LASTEXITCODE -ne 0) {
    Stop-WithError "Could not connect to PostgreSQL using DATABASE_URL."
}


# ------------------------------------------------------------
# 11. Check production model artifacts
# ------------------------------------------------------------

Write-Host ""
Write-Host "Checking production model artifacts..."

$RequiredModels = @(
    "artifacts\models\model_6h.joblib",
    "artifacts\models\model_12h.joblib",
    "artifacts\models\model_24h.joblib"
)

$MissingModels = @()

foreach ($ModelPath in $RequiredModels) {

    $FullModelPath = Join-Path $ProjectRoot $ModelPath

    if (-not (Test-Path $FullModelPath)) {
        $MissingModels += $ModelPath
    }
}

if ($MissingModels.Count -eq 0) {
    Write-Host "[OK] Production model artifacts found."
}
else {

    Write-Host "[WARNING] Some production model artifacts are missing:"

    foreach ($MissingModel in $MissingModels) {
        Write-Host "  - $MissingModel"
    }

    Write-Host ""
    Write-Host "[INFO] You can generate production models using menu option 8."
}


# ------------------------------------------------------------
# 12. Environment ready
# ------------------------------------------------------------

Write-Host ""
Write-Host "============================================================"
Write-Host "ALPHAPULSE ENVIRONMENT READY"
Write-Host "============================================================"
Write-Host ""


# ------------------------------------------------------------
# 13. Show menu
# ------------------------------------------------------------

Write-Host "What do you want to run?"
Write-Host ""
Write-Host "1 - Full analysis + dataset preparation"
Write-Host "2 - Majority baseline"
Write-Host "3 - Logistic Regression"
Write-Host "4 - Random Forest"
Write-Host "5 - XGBoost"
Write-Host "6 - Compare models"
Write-Host "7 - Final test evaluation"
Write-Host "8 - Train production models"
Write-Host "9 - Exit"
Write-Host ""

$choice = Read-Host "Enter option"


# ------------------------------------------------------------
# 14. Run selected stage
# ------------------------------------------------------------

switch ($choice) {

    "1" {
        Write-Host ""
        Write-Host "Running analysis and temporal dataset pipeline..."
        & $VenvPython -m src.analysis.analyze_returns
    }

    "2" {
        Write-Host ""
        Write-Host "Running majority baseline..."
        & $VenvPython -m src.modeling.run_baselines
    }

    "3" {
        Write-Host ""
        Write-Host "Training Logistic Regression..."
        & $VenvPython -m src.modeling.train_logistic_regression
    }

    "4" {
        Write-Host ""
        Write-Host "Training Random Forest..."
        & $VenvPython -m src.modeling.train_random_forest
    }

    "5" {
        Write-Host ""
        Write-Host "Training XGBoost..."
        & $VenvPython -m src.modeling.train_xgboost
    }

    "6" {
        Write-Host ""
        Write-Host "Comparing models..."
        & $VenvPython -m src.modeling.compare_models
    }

    "7" {
        Write-Host ""
        Write-Host "Running final test evaluation..."
        & $VenvPython -m src.modeling.evaluate_final_test
    }

    "8" {
        Write-Host ""
        Write-Host "Training production models..."
        & $VenvPython -m src.modeling.train_production_models
    }

    "9" {
        Write-Host ""
        Write-Host "Exiting."
        exit 0
    }

    default {
        Stop-WithError "Invalid option."
    }
}


# ------------------------------------------------------------
# 15. Validate selected command
# ------------------------------------------------------------

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Program failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[DONE] Command completed successfully."
Write-Host "============================================================"