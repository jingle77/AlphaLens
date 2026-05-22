"""
Main AlphaLens research assistant orchestration layer.

This module coordinates the end-to-end backend workflow:

1. Fetch raw FMP data
2. Clean data
3. Calculate deterministic financial and market metrics
4. Build deterministic evidence package
5. Build prompt payload
6. Generate OpenAI research-style analysis

The Streamlit app should call this module rather than directly calling lower
level data, metric, evidence, prompt, or LLM modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import settings, validate_required_settings
from src.data_cleaner import clean_research_data
from src.data_fetcher import ResearchData, fetch_research_data, normalize_symbol
from src.evidence import SUPPORTED_ANALYSIS_TYPES, build_evidence_package
from src.fmp_client import FMPClient
from src.llm_client import AlphaLensLLMClient, create_llm_client
from src.metrics import calculate_all_metrics
from src.prompts import build_prompt_payload, validate_analysis_type


class ResearchAssistantError(Exception):
    """
    Base exception for AlphaLens research assistant errors.
    """

    pass


@dataclass(frozen=True)
class EquityAnalysisResult:
    """
    Structured result returned by generate_equity_analysis.

    Attributes:
        symbol: Stock ticker analyzed.
        benchmark_symbol: Benchmark ticker, usually SPY.
        analysis_type: Selected AlphaLens analysis type.
        analysis_text: Final LLM-generated research-style analysis.
        financial_metrics: Deterministic financial metrics.
        market_metrics: Deterministic market metrics.
        evidence_package: Structured evidence sent to the prompt layer.
        prompt_payload: Prompt payload sent to the LLM client.
        raw_data: Optional raw FMP data package.
        cleaned_data_shapes: Shape summary for cleaned DataFrames.
    """

    symbol: str
    benchmark_symbol: str
    analysis_type: str
    analysis_text: str
    financial_metrics: dict[str, Any]
    market_metrics: dict[str, Any]
    evidence_package: dict[str, Any]
    prompt_payload: dict[str, Any]
    raw_data: ResearchData | None
    cleaned_data_shapes: dict[str, tuple[int, int]]


def summarize_cleaned_data_shapes(
    cleaned_data: dict[str, Any],
) -> dict[str, tuple[int, int]]:
    """
    Summarize cleaned DataFrame shapes for diagnostics/UI display.

    Args:
        cleaned_data: Dictionary returned by clean_research_data.

    Returns:
        Dictionary mapping dataset name to DataFrame shape.
    """

    shapes: dict[str, tuple[int, int]] = {}

    for name, df in cleaned_data.items():
        shape = getattr(df, "shape", None)

        if shape is None:
            continue

        shapes[name] = tuple(shape)

    return shapes


def generate_equity_analysis(
    symbol: str,
    analysis_type: str,
    benchmark_symbol: str = settings.default_benchmark_symbol,
    statement_limit: int = settings.default_statement_limit,
    news_limit: int = settings.default_news_limit,
    price_history_days: int = settings.default_price_history_days,
    statement_period: str = "annual",
    fmp_client: FMPClient | None = None,
    llm_client: AlphaLensLLMClient | None = None,
    include_raw_data: bool = False,
) -> EquityAnalysisResult:
    """
    Generate an AlphaLens equity analysis for a selected stock and analysis type.

    Args:
        symbol: Stock ticker to analyze.
        analysis_type: Selected analysis type from AlphaLens dropdown.
        benchmark_symbol: Benchmark symbol, usually SPY.
        statement_limit: Number of financial statement periods to fetch.
        news_limit: Number of recent news articles to fetch.
        price_history_days: Approximate calendar-day lookback for prices.
        statement_period: Statement period, usually "annual" or "quarter".
        fmp_client: Optional FMPClient instance, useful for testing.
        llm_client: Optional AlphaLensLLMClient instance, useful for testing.
        include_raw_data: Whether to include raw FMP data in the result.

    Returns:
        EquityAnalysisResult containing text analysis and supporting artifacts.

    Raises:
        ResearchAssistantError: If any pipeline stage fails.
    """

    try:
        validate_required_settings()
        validate_analysis_type(analysis_type)

        clean_symbol = normalize_symbol(symbol)
        clean_benchmark_symbol = normalize_symbol(benchmark_symbol)

        raw_data = fetch_research_data(
            symbol=clean_symbol,
            benchmark_symbol=clean_benchmark_symbol,
            statement_limit=statement_limit,
            news_limit=news_limit,
            price_history_days=price_history_days,
            statement_period=statement_period,
            client=fmp_client,
        )

        cleaned_data = clean_research_data(raw_data)

        metrics = calculate_all_metrics(cleaned_data)

        evidence_package = build_evidence_package(
            symbol=clean_symbol,
            analysis_type=analysis_type,
            cleaned_data=cleaned_data,
            metrics=metrics,
            benchmark_symbol=clean_benchmark_symbol,
        )

        prompt_payload = build_prompt_payload(evidence_package)

        active_llm_client = llm_client or create_llm_client()

        analysis_text = active_llm_client.generate_from_prompt_payload(prompt_payload)

        return EquityAnalysisResult(
            symbol=clean_symbol,
            benchmark_symbol=clean_benchmark_symbol,
            analysis_type=analysis_type,
            analysis_text=analysis_text,
            financial_metrics=metrics["financial"],
            market_metrics=metrics["market"],
            evidence_package=evidence_package,
            prompt_payload=prompt_payload,
            raw_data=raw_data if include_raw_data else None,
            cleaned_data_shapes=summarize_cleaned_data_shapes(cleaned_data),
        )

    except Exception as exc:
        raise ResearchAssistantError(
            f"Failed to generate AlphaLens analysis for symbol='{symbol}', "
            f"analysis_type='{analysis_type}': {exc}"
        ) from exc


def generate_equity_analysis_text(
    symbol: str,
    analysis_type: str,
    benchmark_symbol: str = settings.default_benchmark_symbol,
) -> str:
    """
    Convenience function that returns only the final analysis text.

    Args:
        symbol: Stock ticker to analyze.
        analysis_type: Selected analysis type.
        benchmark_symbol: Benchmark symbol.

    Returns:
        Final research-style analysis text.
    """

    result = generate_equity_analysis(
        symbol=symbol,
        analysis_type=analysis_type,
        benchmark_symbol=benchmark_symbol,
    )

    return result.analysis_text


def get_supported_analysis_types() -> list[str]:
    """
    Return supported analysis types for UI layers.

    Returns:
        List of supported analysis type strings.
    """

    return list(SUPPORTED_ANALYSIS_TYPES)