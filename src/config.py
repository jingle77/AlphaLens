"""
Central configuration for AlphaLens.

This module loads environment variables and exposes a single `settings`
object used throughout the application.

The goal is to keep secrets, API settings, model names, and application
defaults out of the business logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


# Load variables from a local .env file when available.
# In production or hosted environments, these can be supplied directly
# as environment variables instead.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Immutable application settings for AlphaLens.

    Most values are read from environment variables so the app can run
    cleanly in local development, GitHub Codespaces, or a hosted deployment.
    """

    # -----------------------------
    # API keys
    # -----------------------------
    fmp_api_key: str | None = os.getenv("FMP_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    # -----------------------------
    # OpenAI settings
    # -----------------------------
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    # -----------------------------
    # FMP settings
    # -----------------------------
    fmp_base_url: str = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com")

    # Your FMP plan allows 750 requests/minute.
    # We use a safety buffer by default.
    fmp_max_requests_per_minute: int = int(
        os.getenv("FMP_MAX_REQUESTS_PER_MINUTE", "700")
    )

    # -----------------------------
    # AlphaLens defaults
    # -----------------------------
    default_benchmark_symbol: str = os.getenv("DEFAULT_BENCHMARK_SYMBOL", "SPY")

    default_statement_limit: int = int(os.getenv("DEFAULT_STATEMENT_LIMIT", "5"))
    default_news_limit: int = int(os.getenv("DEFAULT_NEWS_LIMIT", "10"))

    # Roughly enough trading/calendar history for 1-year returns plus buffer.
    default_price_history_days: int = int(
        os.getenv("DEFAULT_PRICE_HISTORY_DAYS", "400")
    )

    # Useful starter list for the Streamlit dropdown.
    # Users can still type another ticker manually in app.py later.
    default_ticker_list: tuple[str, ...] = (
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "BRK.B",
        "LLY",
        "JPM",
        "V",
        "UNH",
        "XOM",
        "AVGO",
        "COST",
        "HD",
        "PG",
        "MA",
        "NFLX",
        "AMD",
    )


settings = Settings()


def validate_required_settings() -> None:
    """
    Validate settings required to run the full AlphaLens application.

    This is intentionally not executed at import time. Some tests may import
    settings without needing live API credentials.

    Raises:
        ValueError: If one or more required environment variables are missing.
    """

    missing = []

    if not settings.fmp_api_key:
        missing.append("FMP_API_KEY")

    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")

    if missing:
        missing_vars = ", ".join(missing)
        raise ValueError(
            f"Missing required environment variable(s): {missing_vars}. "
            "Create a local .env file based on .env.example and add your keys."
        )