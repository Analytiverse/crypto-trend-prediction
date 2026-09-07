import time

import requests

from config import (
    API_KEY,
    BASE_URL,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_SECONDS,
)


class CoinGeckoClient:
    def __init__(self):
        if not API_KEY:
            raise ValueError(
                "COINGECKO_API_KEY is missing from the .env file."
            )

        self.session = requests.Session()

        self.session.headers.update(
            {
                "x-cg-demo-api-key": API_KEY
            }
        )

    def get(self, endpoint, params=None):
        url = f"{BASE_URL}{endpoint}"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code == 429:
                    print(
                        f"Rate limit reached. "
                        f"Attempt {attempt}/{MAX_RETRIES}"
                    )

                    if attempt < MAX_RETRIES:
                        wait_time = (
                            RETRY_BACKOFF_SECONDS
                            * attempt
                        )

                        print(
                            f"Waiting {wait_time} seconds "
                            f"before retrying..."
                        )

                        time.sleep(wait_time)

                        continue

                response.raise_for_status()

                return response.json()

            except requests.exceptions.Timeout:
                print(
                    f"Request timed out. "
                    f"Attempt {attempt}/{MAX_RETRIES}"
                )

            except requests.exceptions.ConnectionError:
                print(
                    f"Connection error. "
                    f"Attempt {attempt}/{MAX_RETRIES}"
                )

            except requests.exceptions.HTTPError as exc:
                status_code = None

                if exc.response is not None:
                    status_code = exc.response.status_code

                print(
                    f"HTTP error {status_code}: {exc}. "
                    f"Attempt {attempt}/{MAX_RETRIES}"
                )

                if (
                    status_code is not None
                    and 400 <= status_code < 500
                    and status_code != 429
                ):
                    print(
                        "Non-retryable client error. "
                        "Stopping retries."
                    )

                    break

            except requests.exceptions.RequestException as exc:
                print(
                    f"Request failed: {exc}. "
                    f"Attempt {attempt}/{MAX_RETRIES}"
                )

            if attempt < MAX_RETRIES:
                wait_time = (
                    RETRY_BACKOFF_SECONDS
                    * attempt
                )

                print(
                    f"Waiting {wait_time} seconds "
                    f"before retrying..."
                )

                time.sleep(wait_time)

        print(
            f"Request failed after "
            f"{MAX_RETRIES} attempts: {endpoint}"
        )

        return None