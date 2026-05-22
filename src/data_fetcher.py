"""
Endpoint-specific Financial Modeling Prep data fetchers for AlphaLens.

This module translates AlphaLens research needs into specific FMP endpoint
calls. It does not clean data, calculate metrics, build evidence packages,
or call the LLM.

Raw data returned here is cleaned later by src/data_cleaner.py and transformed
into deterministic metrics by src/metrics.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from src.config import settings
from src.fmp_client import FMPClient, FMPResponseError, create_fmp_client


@dataclass(frozen=True)
class ResearchData:
    """
    Container for raw FMP research data.

    Attributes:
        symbol: Stock ticker being analyzed.
        benchmark_symbol: Benchmark ticker, usually SPY.
        company_profile: Raw company profile data from FMP.
        income_statement: Raw income statement data from FMP.
        balance_sheet: Raw balance sheet data from FMP.
        cash_flow: Raw cash flow statement data from FMP.
        price_history: Raw adjusted price history for the selected stock.
        benchmark_price_history: Raw adjusted price history for benchmark.
        stock_news: Raw recent stock news data.
    """

    symbol: str
    benchmark_symbol: str
    company_profile: list[dict[str, Any]]
    income_statement: list[dict[str, Any]]
    balance_sheet: list[dict[str, Any]]
    cash_flow: list[dict[str, Any]]
    price_history: list[dict[str, Any]]
    benchmark_price_history: list[dict[str, Any]]
    stock_news: list[dict[str, Any]]


def normalize_symbol(symbol: str) -> str:
    """
    Normalize a stock symbol for FMP requests.

    Args:
        symbol: Raw user-entered or dropdown-selected ticker.

    Returns:
        Uppercase ticker string.

    Raises:
        ValueError: If symbol is empty.
    """

    clean_symbol = symbol.strip().upper()

    if not clean_symbol:
        raise ValueError("symbol cannot be empty.")

    return clean_symbol


def get_company_profile(
    symbol: str,
    client: FMPClient | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch company profile data for a stock symbol.

    Args:
        symbol: Stock ticker.
        client: Optional FMP client. If omitted, a default client is created.

    Returns:
        Raw FMP company profile response.
    """

    client = client or create_fmp_client()
    symbol = normalize_symbol(symbol)

    return client.get(
        "/stable/profile",
        params={"symbol": symbol},
    )


def get_income_statement(
    symbol: str,
    limit: int = settings.default_statement_limit,
    period: str = "annual",
    client: FMPClient | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch income statement data for a stock symbol.

    Args:
        symbol: Stock ticker.
        limit: Number of financial statements to request.
        period: Statement period, usually "annual" or "quarter".
        client: Optional FMP client. If omitted, a default client is created.

    Returns:
        Raw FMP income statement response.
    """

    client = client or create_fmp_client()
    symbol = normalize_symbol(symbol)

    return client.get(
        "/stable/income-statement",
        params={
            "symbol": symbol,
            "limit": limit,
            "period": period,
        },
    )


def get_balance_sheet(
    symbol: str,
    limit: int = settings.default_statement_limit,
    period: str = "annual",
    client: FMPClient | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch balance sheet statement data for a stock symbol.

    Args:
        symbol: Stock ticker.
        limit: Number of financial statements to request.
        period: Statement period, usually "annual" or "quarter".
        client: Optional FMP client. If omitted, a default client is created.

    Returns:
        Raw FMP balance sheet response.
    """

    client = client or create_fmp_client()
    symbol = normalize_symbol(symbol)

    return client.get(
        "/stable/balance-sheet-statement",
        params={
            "symbol": symbol,
            "limit": limit,
            "period": period,
        },
    )


def get_cash_flow(
    symbol: str,
    limit: int = settings.default_statement_limit,
    period: str = "annual",
    client: FMPClient | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch cash flow statement data for a stock symbol.

    Args:
        symbol: Stock ticker.
        limit: Number of financial statements to request.
        period: Statement period, usually "annual" or "quarter".
        client: Optional FMP client. If omitted, a default client is created.

    Returns:
        Raw FMP cash flow statement response.
    """

    client = client or create_fmp_client()
    symbol = normalize_symbol(symbol)

    return client.get(
        "/stable/cash-flow-statement",
        params={
            "symbol": symbol,
            "limit": limit,
            "period": period,
        },
    )


def get_price_history(
    symbol: str,
    days: int = settings.default_price_history_days,
    client: FMPClient | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch adjusted end-of-day price history for a stock symbol.

    AlphaLens uses adjusted price data for return calculations so that
    dividends and corporate actions are better reflected in historical
    performance analysis.

    Args:
        symbol: Stock ticker.
        days: Approximate calendar-day lookback window.
        client: Optional FMP client. If omitted, a default client is created.

    Returns:
        Raw FMP adjusted historical price response.
    """

    if days <= 0:
        raise ValueError("days must be greater than 0.")

    client = client or create_fmp_client()
    symbol = normalize_symbol(symbol)

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    return client.get(
        "/stable/historical-price-eod/dividend-adjusted",
        params={
            "symbol": symbol,
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
        },
    )


def get_stock_news(
    symbol: str,
    limit: int = settings.default_news_limit,
    client: FMPClient | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch recent stock news for a symbol.

    News is useful context for the LLM synthesis layer, but it should not be
    allowed to break the whole analysis if FMP has no recent articles for a
    symbol. Therefore, an empty news response returns an empty list.

    Args:
        symbol: Stock ticker.
        limit: Number of news articles to request.
        client: Optional FMP client. If omitted, a default client is created.

    Returns:
        Raw FMP stock news response, or an empty list if no news is available.
    """

    if limit <= 0:
        raise ValueError("limit must be greater than 0.")

    client = client or create_fmp_client()
    symbol = normalize_symbol(symbol)

    try:
        return client.get(
            "/stable/news/stock",
            params={
                "symbols": symbol,
                "limit": limit,
            },
        )
    except FMPResponseError as exc:
        message = str(exc).lower()

        # Treat missing/empty news as non-fatal. Company profile, statements,
        # and price history should remain strict because they are core inputs.
        if "empty list" in message or "no data" in message:
            return []

        raise


def fetch_research_data(
    symbol: str,
    benchmark_symbol: str = settings.default_benchmark_symbol,
    statement_limit: int = settings.default_statement_limit,
    news_limit: int = settings.default_news_limit,
    price_history_days: int = settings.default_price_history_days,
    statement_period: str = "annual",
    client: FMPClient | None = None,
) -> ResearchData:
    """
    Fetch the full raw data package needed for AlphaLens analysis.

    This function intentionally performs no cleaning, metric calculation, or
    LLM prompting. It only collects the raw data needed by later stages.

    Args:
        symbol: Stock ticker to analyze.
        benchmark_symbol: Benchmark symbol, usually SPY.
        statement_limit: Number of financial statements to request.
        news_limit: Number of recent news articles to request.
        price_history_days: Approximate calendar-day price lookback window.
        statement_period: Statement period, usually "annual" or "quarter".
        client: Optional FMP client. If omitted, a default client is created.

    Returns:
        ResearchData object containing raw FMP responses.
    """

    client = client or create_fmp_client()

    symbol = normalize_symbol(symbol)
    benchmark_symbol = normalize_symbol(benchmark_symbol)

    company_profile = get_company_profile(symbol=symbol, client=client)

    income_statement = get_income_statement(
        symbol=symbol,
        limit=statement_limit,
        period=statement_period,
        client=client,
    )

    balance_sheet = get_balance_sheet(
        symbol=symbol,
        limit=statement_limit,
        period=statement_period,
        client=client,
    )

    cash_flow = get_cash_flow(
        symbol=symbol,
        limit=statement_limit,
        period=statement_period,
        client=client,
    )

    price_history = get_price_history(
        symbol=symbol,
        days=price_history_days,
        client=client,
    )

    benchmark_price_history = get_price_history(
        symbol=benchmark_symbol,
        days=price_history_days,
        client=client,
    )

    stock_news = get_stock_news(
        symbol=symbol,
        limit=news_limit,
        client=client,
    )

    return ResearchData(
        symbol=symbol,
        benchmark_symbol=benchmark_symbol,
        company_profile=company_profile,
        income_statement=income_statement,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        price_history=price_history,
        benchmark_price_history=benchmark_price_history,
        stock_news=stock_news,
    )