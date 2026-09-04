import json
import os
from http.server import BaseHTTPRequestHandler

from src.pipelines.daily_market_pipeline import run_daily_pipeline


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        cron_secret = os.getenv("CRON_SECRET")

        if not cron_secret:
            self._send_json(
                status_code=500,
                payload={
                    "status": "error",
                    "message": "CRON_SECRET is not configured.",
                },
            )
            return

        authorization = self.headers.get("Authorization")

        expected_authorization = f"Bearer {cron_secret}"

        if authorization != expected_authorization:
            self._send_json(
                status_code=401,
                payload={
                    "status": "error",
                    "message": "Unauthorized.",
                },
            )
            return

        try:
            result = run_daily_pipeline()

            self._send_json(
                status_code=200,
                payload=result,
            )

        except Exception as exc:
            self._send_json(
                status_code=500,
                payload={
                    "status": "error",
                    "message": str(exc),
                },
            )

    def _send_json(
        self,
        status_code,
        payload,
    ):
        body = json.dumps(
            payload,
            default=str,
        ).encode("utf-8")

        self.send_response(status_code)

        self.send_header(
            "Content-Type",
            "application/json",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()

        self.wfile.write(body)