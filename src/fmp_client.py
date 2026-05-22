"""
Low-level Financial Modeling Prep API client for AlphaLens.

This module handles request mechanics only:
- Session reuse
- API key injection
- Rate limiting
- Retry behavior
- HTTP error handling
- JSON parsing
- Basic response validation

Endpoint-specific finance functions belong in src/data_fetcher.py.
"""

from __future__ import annotations

from typing import Any

import requests
from requests import Response, Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.rate_limiter import RollingWindowRateLimiter, fmp_rate_limiter


class FMPClientError(Exception):
    """
    Base exception for FMP client errors.
    """

    pass


class FMPAuthenticationError(FMPClientError):
    """
    Raised when the FMP API key is missing or rejected.
    """

    pass


class FMPTemporaryError(FMPClientError):
    """
    Raised for temporary API/network errors that may succeed on retry.
    """

    pass


class FMPResponseError(FMPClientError):
    """
    Raised when FMP returns an invalid, empty, or unexpected response.
    """

    pass


class FMPClient:
    """
    Low-level client for the Financial Modeling Prep API.

    Args:
        api_key: FMP API key.
        base_url: Base FMP URL.
        rate_limiter: Rolling-window limiter used before each request.
        timeout_seconds: Request timeout in seconds.
        session: Optional requests.Session, useful for testing.
    """

    def __init__(
        self,
        api_key: str | None = settings.fmp_api_key,
        base_url: str = settings.fmp_base_url,
        rate_limiter: RollingWindowRateLimiter = fmp_rate_limiter,
        timeout_seconds: int = 30,
        session: Session | None = None,
    ) -> None:
        if not api_key:
            raise FMPAuthenticationError(
                "FMP_API_KEY is missing. Add it to your .env file."
            )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.rate_limiter = rate_limiter
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """
        Send a GET request to FMP and return parsed JSON.

        Args:
            endpoint: API endpoint path, for example "/stable/profile".
            params: Optional query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            FMPClientError: For authentication, network, HTTP, JSON, or
                response-validation failures.
        """

        clean_endpoint = self._normalize_endpoint(endpoint)
        url = f"{self.base_url}{clean_endpoint}"

        request_params = dict(params or {})
        request_params["apikey"] = self.api_key

        response = self._request_with_retries(url=url, params=request_params)
        data = self._parse_json(response)
        self._validate_response_payload(data=data, endpoint=clean_endpoint)

        return data

    @retry(
        retry=retry_if_exception_type(FMPTemporaryError),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _request_with_retries(self, url: str, params: dict[str, Any]) -> Response:
        """
        Execute a GET request with retry behavior for temporary failures.

        Temporary failures include:
        - Network/request exceptions
        - HTTP 408
        - HTTP 429
        - HTTP 5xx

        Args:
            url: Fully qualified request URL.
            params: Query parameters including apikey.

        Returns:
            requests.Response object.

        Raises:
            FMPAuthenticationError: For 401/403 responses.
            FMPTemporaryError: For retryable failures.
            FMPClientError: For non-retryable HTTP failures.
        """

        self.rate_limiter.acquire()

        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise FMPTemporaryError(f"Temporary FMP request failure: {exc}") from exc

        if response.status_code in {401, 403}:
            raise FMPAuthenticationError(
                "FMP authentication failed. Check your FMP_API_KEY."
            )

        if response.status_code in {408, 429} or 500 <= response.status_code < 600:
            raise FMPTemporaryError(
                f"Temporary FMP HTTP error {response.status_code}: "
                f"{response.text[:500]}"
            )

        if not response.ok:
            raise FMPClientError(
                f"FMP HTTP error {response.status_code}: {response.text[:500]}"
            )

        return response

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """
        Normalize endpoint into a leading-slash path.

        Args:
            endpoint: Endpoint with or without leading slash.

        Returns:
            Endpoint path with leading slash.
        """

        endpoint = endpoint.strip()

        if not endpoint:
            raise ValueError("endpoint cannot be empty.")

        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"

        return endpoint

    @staticmethod
    def _parse_json(response: Response) -> Any:
        """
        Parse a requests.Response as JSON.

        Args:
            response: HTTP response from FMP.

        Returns:
            Parsed JSON payload.

        Raises:
            FMPResponseError: If JSON parsing fails.
        """

        try:
            return response.json()
        except ValueError as exc:
            raise FMPResponseError(
                f"FMP returned non-JSON response: {response.text[:500]}"
            ) from exc

    @staticmethod
    def _validate_response_payload(data: Any, endpoint: str) -> None:
        """
        Validate that the FMP response payload is usable.

        FMP usually returns lists or dictionaries. Empty lists are possible for
        invalid symbols or unavailable endpoints, so we raise a clear error.

        Args:
            data: Parsed JSON payload.
            endpoint: Endpoint used for the request.

        Raises:
            FMPResponseError: If the payload is empty or indicates an error.
        """

        if data is None:
            raise FMPResponseError(f"FMP returned no data for endpoint: {endpoint}")

        if isinstance(data, list) and len(data) == 0:
            raise FMPResponseError(f"FMP returned an empty list for endpoint: {endpoint}")

        if isinstance(data, dict):
            possible_error_keys = {"Error Message", "error", "message"}

            for key in possible_error_keys:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    raise FMPResponseError(
                        f"FMP returned an error for endpoint {endpoint}: {value}"
                    )

            if len(data) == 0:
                raise FMPResponseError(
                    f"FMP returned an empty dictionary for endpoint: {endpoint}"
                )


def create_fmp_client() -> FMPClient:
    """
    Factory function for creating a configured FMP client.

    Keeping this as a factory avoids forcing live API credentials during simple
    module imports in tests or documentation workflows.

    Returns:
        Configured FMPClient instance.
    """

    return FMPClient()