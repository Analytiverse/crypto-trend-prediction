# ============================================================
# AlphaPulse - Quick Start
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================"
Write-Host "        ALPHAPULSE - CRYPTO TREND PREDICTION"
Write-Host "============================================================"
Write-Host ""

# ------------------------------------------------------------
# 1. Move automatically to project root
# ------------------------------------------------------------

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "[OK] Project directory: $ProjectRoot"


# ------------------------------------------------------------
# 2. Activate virtual environment
# ------------------------------------------------------------

$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $VenvActivate)) {
    Write-Host ""
    Write-Host "[ERROR] .venv was not found."
    Write-Host "Expected: $VenvActivate"
    exit 1
}

& $VenvActivate

Write-Host "[OK] Virtual environment activated."


# ------------------------------------------------------------
# 3. Show menu
# ------------------------------------------------------------

Write-Host ""
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
# 4. Run selected stage
# ------------------------------------------------------------

switch ($choice) {

    "1" {
        Write-Host ""
        Write-Host "Running analysis and temporal dataset pipeline..."
        python -m src.analysis.analyze_returns
    }

    "2" {
        Write-Host ""
        Write-Host "Running majority baseline..."
        python -m src.modeling.run_baselines
    }

    "3" {
        Write-Host ""
        Write-Host "Training Logistic Regression..."
        python -m src.modeling.train_logistic_regression
    }

    "4" {
        Write-Host ""
        Write-Host "Training Random Forest..."
        python -m src.modeling.train_random_forest
    }

    "5" {
        Write-Host ""
        Write-Host "Training XGBoost..."
        python -m src.modeling.train_xgboost
    }

    "6" {
        Write-Host ""
        Write-Host "Comparing models..."
        python -m src.modeling.compare_models
    }

    "7" {
        Write-Host ""
        Write-Host "Running final test evaluation..."
        python -m src.modeling.evaluate_final_test
    }

    "8" {
        Write-Host ""
        Write-Host "Training production models..."
        python -m src.modeling.train_production_models
    }

    "9" {
        Write-Host ""
        Write-Host "Exiting."
        exit
    }

    default {
        Write-Host ""
        Write-Host "[ERROR] Invalid option."
        exit 1
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Program failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[DONE] Command completed successfully."
Write-Host "============================================================"