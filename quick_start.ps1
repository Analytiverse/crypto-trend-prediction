# ============================================================
# AlphaPulse - Quick Start
# Windows launcher
# ============================================================

$ErrorActionPreference = "Stop"


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

Write-Host ""
Write-Host "============================================================"
Write-Host "        ALPHAPULSE - CRYPTO TREND PREDICTION"
Write-Host "============================================================"
Write-Host ""

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
# 7. Install project dependencies
# ------------------------------------------------------------

Write-Host ""
Write-Host "Checking project dependencies..."

& $VenvPython -m pip install -r $RequirementsFile

if ($LASTEXITCODE -ne 0) {
    Stop-WithError "Dependency installation failed. Check requirements.txt and your internet connection."
}

Write-Host "[OK] Project dependencies are installed."


# ------------------------------------------------------------
# 8. Run shared AlphaPulse bootstrap
# ------------------------------------------------------------

Write-Host ""
Write-Host "Running shared AlphaPulse bootstrap..."

$SharedBootstrap = Join-Path $ProjectRoot "src\bootstrap.py"

if (-not (Test-Path $SharedBootstrap)) {
    Stop-WithError "Shared bootstrap module was not found: $SharedBootstrap"
}

& $VenvPython -m src.bootstrap

if ($LASTEXITCODE -ne 0) {
    Stop-WithError "Shared AlphaPulse bootstrap failed."
}


# ------------------------------------------------------------
# 9. Environment ready
# ------------------------------------------------------------

Write-Host ""
Write-Host "============================================================"
Write-Host "ALPHAPULSE WINDOWS ENVIRONMENT READY"
Write-Host "============================================================"
Write-Host ""


# ------------------------------------------------------------
# 10. Show menu
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
# 11. Run selected stage
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
        & $VenvPython -m src.ml.evaluation.run_baselines
    }

    "3" {
        Write-Host ""
        Write-Host "Training Logistic Regression..."
        & $VenvPython -m src.ml.training.train_logistic_regression
    }

    "4" {
        Write-Host ""
        Write-Host "Training Random Forest..."
        & $VenvPython -m src.ml.training.train_random_forest
    }

    "5" {
        Write-Host ""
        Write-Host "Training XGBoost..."
        & $VenvPython -m src.ml.training.train_xgboost
    }

    "6" {
        Write-Host ""
        Write-Host "Comparing models..."
        & $VenvPython -m src.ml.evaluation.compare_models
    }

    "7" {
        Write-Host ""
        Write-Host "Running final test evaluation..."
        & $VenvPython -m src.ml.evaluation.evaluate_final_test
    }

    "8" {
        Write-Host ""
        Write-Host "Training production models..."
        & $VenvPython -m src.ml.training.train_production_models
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
# 12. Validate selected command
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