import json
from decimal import Decimal
from http.server import BaseHTTPRequestHandler

from sqlalchemy import text

from src.database.connection import get_engine


def serialize_value(value):
    if isinstance(value, Decimal):
        return float(value)

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return value


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            engine = get_engine()

            with engine.connect() as connection:
                latest_rows = connection.execute(
                    text(
                        """
                        SELECT DISTINCT ON (mh.coin_id)
                            mh.coin_id,
                            c.symbol,
                            c.name,
                            mh.price,
                            mh.market_cap,
                            mh.total_volume,
                            mh.timestamp
                        FROM market_hourly mh
                        JOIN coins c
                            ON c.coin_id = mh.coin_id
                        ORDER BY
                            mh.coin_id,
                            mh.timestamp DESC;
                        """
                    )
                ).mappings().all()

                recent_rows = connection.execute(
                    text(
                        """
                        SELECT
                            mh.coin_id,
                            c.symbol,
                            c.name,
                            mh.price,
                            mh.market_cap,
                            mh.total_volume,
                            mh.timestamp
                        FROM market_hourly mh
                        JOIN coins c
                            ON c.coin_id = mh.coin_id
                        ORDER BY mh.timestamp DESC
                        LIMIT 25;
                        """
                    )
                ).mappings().all()

            payload = {
                "status": "success",
                "latest": [
                    {
                        key: serialize_value(value)
                        for key, value in row.items()
                    }
                    for row in latest_rows
                ],
                "recent": [
                    {
                        key: serialize_value(value)
                        for key, value in row.items()
                    }
                    for row in recent_rows
                ],
            }

            self._send_json(200, payload)

        except Exception as exc:
            self._send_json(
                500,
                {
                    "status": "error",
                    "message": str(exc),
                },
            )

    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")

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