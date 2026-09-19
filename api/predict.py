"""
AlphaPulse Prediction API

Endpoint:

    GET /api/predict?asset=BTC&horizon=6

Supported assets:
    BTC
    ETH
    SOL
    XRP
    ADA

Supported horizons:
    6
    12
    24

The numerical prediction comes entirely from the trained
AlphaPulse machine-learning model.

No LLM is involved in generating the numerical prediction.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from src.services.prediction_service import generate_prediction


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_HORIZONS = {6, 12, 24}

SUPPORTED_ASSETS = {
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "ADA",
    "BITCOIN",
    "ETHEREUM",
    "SOLANA",
    "RIPPLE",
    "CARDANO",
}


# ============================================================
# JSON RESPONSE HELPER
# ============================================================

def json_response(
    handler: BaseHTTPRequestHandler,
    status_code: int,
    payload: dict,
) -> None:
    """
    Send a JSON HTTP response.
    """

    body = json.dumps(
        payload,
        indent=2,
        default=str,
    ).encode("utf-8")

    handler.send_response(status_code)

    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8",
    )

    handler.send_header(
        "Access-Control-Allow-Origin",
        "*",
    )

    handler.send_header(
        "Access-Control-Allow-Methods",
        "GET, OPTIONS",
    )

    handler.send_header(
        "Access-Control-Allow-Headers",
        "Content-Type, Authorization",
    )

    handler.send_header(
        "Cache-Control",
        "no-store",
    )

    handler.send_header(
        "Content-Length",
        str(len(body)),
    )

    handler.end_headers()

    handler.wfile.write(body)


# ============================================================
# REQUEST VALIDATION
# ============================================================

def get_single_query_parameter(
    query_params: dict,
    parameter_name: str,
) -> str | None:
    """
    Return the first value for a query-string parameter.
    """

    values = query_params.get(parameter_name)

    if not values:
        return None

    value = values[0]

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def validate_asset(
    asset: str | None,
) -> str:
    """
    Validate the requested cryptocurrency.
    """

    if asset is None:
        raise ValueError(
            "Missing required query parameter: asset. "
            "Example: /api/predict?asset=BTC&horizon=6"
        )

    normalized = asset.strip().upper()

    if normalized not in SUPPORTED_ASSETS:
        raise ValueError(
            f"Unsupported asset '{asset}'. "
            "Supported assets are BTC, ETH, SOL, XRP and ADA."
        )

    return normalized


def validate_horizon(
    horizon_value: str | None,
) -> int:
    """
    Validate the prediction horizon.
    """

    if horizon_value is None:
        raise ValueError(
            "Missing required query parameter: horizon. "
            "Supported horizons are 6, 12 and 24 hours."
        )

    try:
        horizon = int(horizon_value)
    except (TypeError, ValueError):
        raise ValueError(
            "horizon must be an integer: 6, 12 or 24."
        )

    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(
            f"Unsupported horizon '{horizon}'. "
            "Supported horizons are 6, 12 and 24 hours."
        )

    return horizon


# ============================================================
# API HANDLER
# ============================================================

class handler(BaseHTTPRequestHandler):
    """
    Vercel Python serverless-function handler.
    """

    def do_OPTIONS(self) -> None:
        """
        Handle browser CORS preflight requests.
        """

        self.send_response(204)

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, OPTIONS",
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization",
        )

        self.end_headers()

    def do_GET(self) -> None:
        """
        Generate an AlphaPulse trend prediction.
        """

        try:
            parsed_url = urlparse(
                self.path
            )

            query_params = parse_qs(
                parsed_url.query
            )

            asset_parameter = (
                get_single_query_parameter(
                    query_params,
                    "asset",
                )
            )

            horizon_parameter = (
                get_single_query_parameter(
                    query_params,
                    "horizon",
                )
            )

            asset = validate_asset(
                asset_parameter
            )

            horizon = validate_horizon(
                horizon_parameter
            )

            prediction = generate_prediction(
                asset=asset,
                horizon_hours=horizon,
      )

            response = {
                "success": True,
                "prediction": prediction,
            }

            json_response(
                handler=self,
                status_code=200,
                payload=response,
            )

        except ValueError as exc:

            json_response(
                handler=self,
                status_code=400,
                payload={
                    "success": False,
                    "error": "Invalid request",
                    "message": str(exc),
                },
            )

        except Exception as exc:

            print(
                "Prediction API error:",
                repr(exc),
            )

            json_response(
                handler=self,
                status_code=500,
                payload={
                    "success": False,
                    "error": "Prediction failed",
                    "message": str(exc),
                },
            )

    def log_message(
        self,
        format: str,
        *args,
    ) -> None:
        """
        Preserve useful request logging.
        """

        print(
            "%s - - [%s] %s"
            % (
                self.address_string(),
                self.log_date_time_string(),
                format % args,
            )
        )