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
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    Write-Host ""
    exit 1
}


# ------------------------------------------------------------
# Helper: securely request a missing environment variable
# ------------------------------------------------------------

function Read-SecretValue {
    param(
        [string]$VariableName
    )

    $SecureValue = Read-Host "Enter value for $VariableName" -AsSecureString

    $BSTR = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureValue)

    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($BSTR)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)
    }
}


# ------------------------------------------------------------
# Helper: read variable from .env
# ------------------------------------------------------------

function Get-DotEnvValue {
    param(
        [string]$EnvFile,
        [string]$VariableName
    )

    if (-not (Test-Path $EnvFile)) {
        return $null
    }

    foreach ($Line in Get-Content $EnvFile) {

        $TrimmedLine = $Line.Trim()

        if ([string]::IsNullOrWhiteSpace($TrimmedLine)) {
            continue
        }

        if ($TrimmedLine.StartsWith("#")) {
            continue
        }

        $Parts = $TrimmedLine -split "=", 2

        if ($Parts.Count -ne 2) {
            continue
        }

        $Name = $Parts[0].Trim()

        if ($Name -ne $VariableName) {
            continue
        }

        $Value = $Parts[1].Trim()

        # Remove matching surrounding quotes
        if ($Value.Length -ge 2) {

            if (
                ($Value.StartsWith('"') -and $Value.EndsWith('"')) -or
                ($Value.StartsWith("'") -and $Value.EndsWith("'"))
            ) {
                $Value = $Value.Substring(1, $Value.Length - 2)
            }
        }

        return $Value
    }

    return $null
}


# ------------------------------------------------------------
# Helper: safely add/update variable in .env
# ------------------------------------------------------------

function Set-DotEnvValue {
    param(
        [string]$EnvFile,
        [string]$VariableName,
        [string]$VariableValue
    )

    $Lines = @()

    if (Test-Path $EnvFile) {
        $Lines = @(Get-Content $EnvFile)
    }

    $UpdatedLines = @()
    $Found = $false

    foreach ($Line in $Lines) {

        $TrimmedLine = $Line.Trim()

        if (
            -not $TrimmedLine.StartsWith("#") -and
            $TrimmedLine -match "^\s*$([regex]::Escape($VariableName))\s*="
        ) {
            $UpdatedLines += "$VariableName=$VariableValue"
            $Found = $true
        }
        else {
            $UpdatedLines += $Line
        }
    }

    if (-not $Found) {

        if (
            $UpdatedLines.Count -gt 0 -and
            -not [string]::IsNullOrWhiteSpace($UpdatedLines[-1])
        ) {
            $UpdatedLines += ""
        }

        $UpdatedLines += "$VariableName=$VariableValue"
    }

    Set-Content `
        -Path $EnvFile `
        -Value $UpdatedLines `
        -Encoding UTF8
}


# ------------------------------------------------------------
# Helper: detect placeholder values
# ------------------------------------------------------------

function Test-PlaceholderValue {
    param(
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $true
    }

    $Normalized = $Value.Trim().ToLower()

    $Placeholders = @(
        "your_api_key",
        "your-api-key",
        "your_key",
        "your-key",
        "changeme",
        "change_me",
        "replace_me",
        "replace-me",
        "placeholder",
        "<your_api_key>",
        "<your_database_url>",
        "<your_cron_secret>",
        "<your_groq_api_key>"
    )

    return $Placeholders -contains $Normalized
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
# 3. Validate Python >= 3.10
# ------------------------------------------------------------

Write-Host "Checking Python version compatibility..."

if ($PythonCommand -eq "py") {

    $VersionCheck = & py -c `
        "import sys; print(1 if sys.version_info >= (3,10) else 0)"

}
else {

    $VersionCheck = & python -c `
        "import sys; print(1 if sys.version_info >= (3,10) else 0)"
}


if ($VersionCheck -ne "1") {
    Stop-WithError "AlphaPulse requires Python 3.10 or newer. Detected: $PythonVersion"
}

Write-Host "[OK] Python version is compatible."


# ------------------------------------------------------------
# 4. Check important project files
# ------------------------------------------------------------

Write-Host ""
Write-Host "Checking project files..."

$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
$SharedBootstrap = Join-Path $ProjectRoot "src\bootstrap.py"

if (-not (Test-Path $RequirementsFile)) {
    Stop-WithError "requirements.txt was not found in the project root."
}

if (-not (Test-Path $SharedBootstrap)) {
    Stop-WithError "Shared bootstrap module was not found: $SharedBootstrap"
}

Write-Host "[OK] requirements.txt found."
Write-Host "[OK] src/bootstrap.py found."


# ------------------------------------------------------------
# 5. Create virtual environment when missing
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
# 6. Activate virtual environment
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
    Write-Host "[INFO] AlphaPulse will continue using .venv Python directly."
}

Write-Host "[OK] Virtual environment ready."


# ------------------------------------------------------------
# 7. Check pip
# ------------------------------------------------------------

Write-Host ""
Write-Host "Checking pip..."

& $VenvPython -m pip --version

if ($LASTEXITCODE -ne 0) {
    Stop-WithError "pip is not available inside the virtual environment."
}

Write-Host "[OK] pip is available."


# ------------------------------------------------------------
# 8. Upgrade pip
# ------------------------------------------------------------

Write-Host ""
Write-Host "Checking/upgrading pip..."

& $VenvPython -m pip install --upgrade pip

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] pip upgrade failed. Continuing with the installed pip version."
}
else {
    Write-Host "[OK] pip is up to date."
}


# ------------------------------------------------------------
# 9. Install/sync project dependencies
# ------------------------------------------------------------

Write-Host ""
Write-Host "Installing/syncing project dependencies..."

& $VenvPython -m pip install -r $RequirementsFile

if ($LASTEXITCODE -ne 0) {
    Stop-WithError "Dependency installation failed. Check requirements.txt and your internet connection."
}

Write-Host "[OK] Project dependencies are installed."


# ------------------------------------------------------------
# 10. Prepare .env
# ------------------------------------------------------------

Write-Host ""
Write-Host "Checking environment configuration..."

$EnvFile = Join-Path $ProjectRoot ".env"
$EnvExampleFile = Join-Path $ProjectRoot ".env.example"


if (-not (Test-Path $EnvFile)) {

    Write-Host "[INFO] .env file was not found."

    if (Test-Path $EnvExampleFile) {

        Write-Host "[INFO] Creating .env from .env.example..."

        Copy-Item `
            -Path $EnvExampleFile `
            -Destination $EnvFile

        Write-Host "[OK] .env created from .env.example."
    }
    else {

        Write-Host "[INFO] .env.example was not found."
        Write-Host "[INFO] Creating empty .env..."

        New-Item `
            -Path $EnvFile `
            -ItemType File `
            -Force | Out-Null

        Write-Host "[OK] Empty .env created."
    }
}
else {
    Write-Host "[OK] .env file found."
}


# ------------------------------------------------------------
# 11. Check .gitignore protection
# ------------------------------------------------------------

$GitIgnoreFile = Join-Path $ProjectRoot ".gitignore"

if (Test-Path $GitIgnoreFile) {

    $GitIgnoreContent = Get-Content $GitIgnoreFile

    $EnvIgnored = $false

    foreach ($Line in $GitIgnoreContent) {

        $Trimmed = $Line.Trim()

        if (
            $Trimmed -eq ".env" -or
            $Trimmed -eq "/.env"
        ) {
            $EnvIgnored = $true
            break
        }
    }

    if ($EnvIgnored) {
        Write-Host "[OK] .env is protected by .gitignore."
    }
    else {
        Write-Host "[WARNING] .env is not listed in .gitignore."
        Write-Host "[WARNING] Add '.env' to .gitignore before committing."
    }
}
else {
    Write-Host "[WARNING] .gitignore was not found."
    Write-Host "[WARNING] Make sure .env is never committed."
}


# ------------------------------------------------------------
# 12. Check / assign required environment variables
# ------------------------------------------------------------

Write-Host ""
Write-Host "Checking required environment variables..."


$RequiredEnvVars = @(
    "COINGECKO_API_KEY",
    "DATABASE_URL",
    "CRON_SECRET",
    "GROQ_API_KEY"
)


foreach ($VariableName in $RequiredEnvVars) {

    # --------------------------------------------------------
    # Priority 1: already assigned to current process / OS
    # --------------------------------------------------------

    $CurrentValue = [Environment]::GetEnvironmentVariable(
        $VariableName,
        "Process"
    )


    if (
        -not [string]::IsNullOrWhiteSpace($CurrentValue) -and
        -not (Test-PlaceholderValue $CurrentValue)
    ) {

        Write-Host "[OK] $VariableName already assigned in environment."
        continue
    }


    # --------------------------------------------------------
    # Priority 2: value available in .env
    # --------------------------------------------------------

    $DotEnvValue = Get-DotEnvValue `
        -EnvFile $EnvFile `
        -VariableName $VariableName


    if (
        -not [string]::IsNullOrWhiteSpace($DotEnvValue) -and
        -not (Test-PlaceholderValue $DotEnvValue)
    ) {

        [Environment]::SetEnvironmentVariable(
            $VariableName,
            $DotEnvValue,
            "Process"
        )

        Write-Host "[OK] $VariableName loaded from .env and assigned."

        continue
    }


    # --------------------------------------------------------
    # Priority 3: ask once and persist
    # --------------------------------------------------------

    Write-Host ""
    Write-Host "[INFO] $VariableName is not configured."
    Write-Host "[INFO] Enter it once. It will be saved in .env."


    $EnteredValue = Read-SecretValue `
        -VariableName $VariableName


    if (
        [string]::IsNullOrWhiteSpace($EnteredValue) -or
        (Test-PlaceholderValue $EnteredValue)
    ) {
        Stop-WithError "$VariableName must contain a valid value."
    }


    # Basic DATABASE_URL validation
    if ($VariableName -eq "DATABASE_URL") {

        if (
            -not $EnteredValue.StartsWith("postgresql://") -and
            -not $EnteredValue.StartsWith("postgres://")
        ) {
            Stop-WithError "DATABASE_URL must start with postgresql:// or postgres://"
        }
    }


    Set-DotEnvValue `
        -EnvFile $EnvFile `
        -VariableName $VariableName `
        -VariableValue $EnteredValue


    [Environment]::SetEnvironmentVariable(
        $VariableName,
        $EnteredValue,
        "Process"
    )


    Write-Host "[OK] $VariableName saved to .env and assigned."
}


# ------------------------------------------------------------
# 13. Final environment verification
# ------------------------------------------------------------

Write-Host ""
Write-Host "Verifying environment variables..."


foreach ($VariableName in $RequiredEnvVars) {

    $FinalValue = [Environment]::GetEnvironmentVariable(
        $VariableName,
        "Process"
    )

    if (
        [string]::IsNullOrWhiteSpace($FinalValue) -or
        (Test-PlaceholderValue $FinalValue)
    ) {
        Stop-WithError "$VariableName is still missing after environment setup."
    }

    Write-Host "[OK] $VariableName ready."
}


Write-Host "[OK] All required environment variables are assigned."


# ------------------------------------------------------------
# 14. Run shared AlphaPulse bootstrap
# ------------------------------------------------------------

Write-Host ""
Write-Host "Running shared AlphaPulse bootstrap..."

& $VenvPython -m src.bootstrap

if ($LASTEXITCODE -ne 0) {
    Stop-WithError "Shared AlphaPulse bootstrap failed."
}


# ------------------------------------------------------------
# 15. Environment ready
# ------------------------------------------------------------

Write-Host ""
Write-Host "============================================================"
Write-Host "ALPHAPULSE WINDOWS ENVIRONMENT READY"
Write-Host "============================================================"
Write-Host ""


# ------------------------------------------------------------
# 16. Show menu
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
# 17. Run selected stage
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
# 18. Validate selected command
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