import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

REQUIRED_ENV_VARS = [
    "DATABASE_URL",
    "COINGECKO_API_KEY",
]

REQUIRED_MODELS = [
    PROJECT_ROOT / "artifacts" / "models" / "model_6h.joblib",
    PROJECT_ROOT / "artifacts" / "models" / "model_12h.joblib",
    PROJECT_ROOT / "artifacts" / "models" / "model_24h.joblib",
]


def print_header():
    print("")
    print("============================================================")
    print("ALPHAPULSE SHARED BOOTSTRAP")
    print("============================================================")
    print("")


def validate_environment():
    print("Checking environment configuration...")

    if not ENV_FILE.exists():
        raise FileNotFoundError(
            f".env file was not found: {ENV_FILE}"
        )

    load_dotenv(
        dotenv_path=ENV_FILE,
        override=False,
    )

    print("[OK] .env file found.")

    missing = [
        name
        for name in REQUIRED_ENV_VARS
        if not os.getenv(name)
    ]

    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
        )

    print("[OK] DATABASE_URL is configured.")
    print("[OK] COINGECKO_API_KEY is configured.")

    if os.getenv("GROQ_API_KEY"):
        print("[OK] GROQ_API_KEY is configured.")
    else:
        print(
            "[INFO] GROQ_API_KEY is not configured. "
            "AI explanations may be unavailable."
        )


def test_database_connection():
    print("")
    print("Testing PostgreSQL connection...")

    from src.infrastructure.database.connection import get_engine

    engine = get_engine()

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    print("[OK] PostgreSQL connection successful.")


def run_database_bootstrap():
    print("")
    print("Checking AlphaPulse database bootstrap...")

    from src.database.bootstrap_database import bootstrap_database

    result = bootstrap_database()

    if result["seeded"]:
        print(
            "[OK] Fresh PostgreSQL database was initialized "
            "using the bundled seed."
        )
    else:
        print(
            "[OK] Existing PostgreSQL market data is ready."
        )


def check_model_artifacts():
    print("")
    print("Checking production model artifacts...")

    missing = [
        model
        for model in REQUIRED_MODELS
        if not model.exists()
    ]

    if not missing:
        print("[OK] Production model artifacts found.")
        return True

    print(
        "[WARNING] Some production model artifacts are missing:"
    )

    for model in missing:
        try:
            relative_path = model.relative_to(PROJECT_ROOT)
        except ValueError:
            relative_path = model

        print(f"  - {relative_path}")

    print(
        "[INFO] Production models must be generated "
        "before prediction functionality is available."
    )

    return False


def run_bootstrap():
    print_header()

    validate_environment()

    test_database_connection()

    run_database_bootstrap()

    models_available = check_model_artifacts()

    print("")
    print("============================================================")

    if models_available:
        print("ALPHAPULSE SHARED BOOTSTRAP READY")
    else:
        print(
            "ALPHAPULSE BOOTSTRAP READY "
            "(PRODUCTION MODELS MISSING)"
        )

    print("============================================================")
    print("")

    return {
        "environment": True,
        "database": True,
        "models": models_available,
    }


def main():
    try:
        run_bootstrap()

    except Exception as exc:
        print("")
        print("============================================================")
        print("[ERROR] ALPHAPULSE BOOTSTRAP FAILED")
        print("============================================================")
        print(str(exc))
        print("")

        sys.exit(1)


if __name__ == "__main__":
    main()