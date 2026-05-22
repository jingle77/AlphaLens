"""
Deterministic evidence routing for AlphaLens.

This module assembles prompt-specific evidence packages from cleaned FMP data
and deterministic metrics.

Important:
This is not semantic RAG. There are no embeddings, vector databases, or dynamic
similarity searches. The selected analysis type deterministically controls which
evidence fields are included for LLM synthesis.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


ANALYSIS_OVERALL = "Overall outperformance thesis"
ANALYSIS_BULL = "Bull case for outperformance"
ANALYSIS_BEAR = "Bear case against outperformance"
ANALYSIS_QUALITY = "Financial quality assessment"
ANALYSIS_IMPROVEMENT = "What would need to improve?"


SUPPORTED_ANALYSIS_TYPES = (
    ANALYSIS_OVERALL,
    ANALYSIS_BULL,
    ANALYSIS_BEAR,
    ANALYSIS_QUALITY,
    ANALYSIS_IMPROVEMENT,
)


def serialize_value(value: Any) -> Any:
    """
    Convert pandas/numpy/python values into JSON-safe values.

    Args:
        value: Arbitrary value from DataFrames or metric dictionaries.

    Returns:
        JSON-safe Python value.
    """

    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.date().isoformat()

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if isinstance(value, dict):
        return {key: serialize_value(val) for key, val in value.items()}

    if isinstance(value, list):
        return [serialize_value(item) for item in value]

    if isinstance(value, tuple):
        return [serialize_value(item) for item in value]

    return value


def compact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """
    Remove keys with None values from a dictionary.

    Args:
        data: Dictionary to compact.

    Returns:
        Dictionary with None values removed.
    """

    return {key: serialize_value(value) for key, value in data.items() if value is not None}


def get_first_row_as_dict(df: pd.DataFrame | None) -> dict[str, Any]:
    """
    Return the first row of a DataFrame as a dictionary.

    Args:
        df: Input DataFrame.

    Returns:
        First-row dictionary, or empty dictionary if unavailable.
    """

    if df is None or df.empty:
        return {}

    return df.iloc[0].to_dict()


def get_company_overview(cleaned_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Build a compact company overview from the cleaned profile DataFrame.

    Args:
        cleaned_data: Dictionary returned by clean_research_data.

    Returns:
        Company overview dictionary.
    """

    profile = get_first_row_as_dict(cleaned_data.get("company_profile"))

    return compact_dict(
        {
            "symbol": profile.get("symbol"),
            "company_name": profile.get("companyName") or profile.get("company_name"),
            "sector": profile.get("sector"),
            "industry": profile.get("industry"),
            "exchange": profile.get("exchange"),
            "market_cap": profile.get("marketCap"),
            "beta": profile.get("beta"),
            "price": profile.get("price"),
            "description": truncate_text(
                profile.get("description") or profile.get("companyDescription"),
                max_chars=700,
            ),
        }
    )


def truncate_text(text: Any, max_chars: int = 500) -> str | None:
    """
    Truncate long text fields to keep evidence packages compact.

    Args:
        text: Text-like value.
        max_chars: Maximum number of characters.

    Returns:
        Truncated string or None.
    """

    if text is None:
        return None

    text = str(text).strip()

    if not text:
        return None

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3].rstrip() + "..."


def get_financial_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Extract core financial metrics.

    Args:
        metrics: Dictionary returned by calculate_all_metrics.

    Returns:
        Financial snapshot dictionary.
    """

    financial = metrics.get("financial", {})

    return compact_dict(
        {
            "latest_fiscal_date": financial.get("latest_fiscal_date"),
            "revenue": financial.get("revenue"),
            "prior_revenue": financial.get("prior_revenue"),
            "revenue_growth": financial.get("revenue_growth"),
            "gross_margin": financial.get("gross_margin"),
            "operating_margin": financial.get("operating_margin"),
            "net_margin": financial.get("net_margin"),
            "operating_cash_flow": financial.get("operating_cash_flow"),
            "free_cash_flow": financial.get("free_cash_flow"),
            "free_cash_flow_margin": financial.get("free_cash_flow_margin"),
            "cash_and_equivalents": financial.get("cash_and_equivalents"),
            "total_debt": financial.get("total_debt"),
            "total_equity": financial.get("total_equity"),
            "debt_to_equity": financial.get("debt_to_equity"),
        }
    )


def get_market_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Extract core market and benchmark-relative metrics.

    Args:
        metrics: Dictionary returned by calculate_all_metrics.

    Returns:
        Market snapshot dictionary.
    """

    market = metrics.get("market", {})

    return compact_dict(
        {
            "price_start_date": market.get("price_start_date"),
            "price_end_date": market.get("price_end_date"),
            "latest_stock_price": market.get("latest_stock_price"),
            "latest_benchmark_price": market.get("latest_benchmark_price"),
            "return_windows": market.get("return_windows"),
            "annualized_volatility": market.get("annualized_volatility"),
            "benchmark_annualized_volatility": market.get(
                "benchmark_annualized_volatility"
            ),
            "max_drawdown": market.get("max_drawdown"),
            "benchmark_max_drawdown": market.get("benchmark_max_drawdown"),
        }
    )


def get_recent_news(
    cleaned_data: dict[str, pd.DataFrame],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Extract compact recent news items.

    Args:
        cleaned_data: Dictionary returned by clean_research_data.
        limit: Maximum number of news items.

    Returns:
        List of compact news dictionaries.
    """

    news_df = cleaned_data.get("stock_news")

    if news_df is None or news_df.empty:
        return []

    items: list[dict[str, Any]] = []

    for _, row in news_df.head(limit).iterrows():
        items.append(
            compact_dict(
                {
                    "published_date": row.get("publishedDate")
                    or row.get("date")
                    or row.get("published_date"),
                    "title": row.get("title"),
                    "source": row.get("site") or row.get("publisher") or row.get("source"),
                    "summary_excerpt": truncate_text(
                        row.get("text")
                        or row.get("summary")
                        or row.get("content")
                        or row.get("description"),
                        max_chars=350,
                    ),
                    "url": row.get("url"),
                }
            )
        )

    return items


def format_percent(value: Any) -> str:
    """
    Format decimal metrics as human-readable percentages.

    Args:
        value: Decimal value, for example 0.123.

    Returns:
        Human-readable percentage string or unavailable marker.
    """

    if value is None:
        return "unavailable"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unavailable"

    if np.isnan(value) or np.isinf(value):
        return "unavailable"

    return f"{value:.1%}"


def classify_financial_signals(financial: dict[str, Any]) -> dict[str, list[str]]:
    """
    Create deterministic financial signal labels.

    These are intentionally simple heuristics. They are not recommendations.
    Their purpose is to help the LLM weigh evidence consistently.

    Args:
        financial: Financial snapshot dictionary.

    Returns:
        Dictionary with supports, weakens, and uncertainties lists.
    """

    supports: list[str] = []
    weakens: list[str] = []
    uncertainties: list[str] = []

    revenue_growth = financial.get("revenue_growth")
    gross_margin = financial.get("gross_margin")
    operating_margin = financial.get("operating_margin")
    net_margin = financial.get("net_margin")
    free_cash_flow = financial.get("free_cash_flow")
    free_cash_flow_margin = financial.get("free_cash_flow_margin")
    debt_to_equity = financial.get("debt_to_equity")

    if revenue_growth is None:
        uncertainties.append("Revenue growth could not be calculated.")
    elif revenue_growth > 0:
        supports.append(f"Revenue grew year over year ({format_percent(revenue_growth)}).")
    else:
        weakens.append(f"Revenue declined year over year ({format_percent(revenue_growth)}).")

    if gross_margin is None:
        uncertainties.append("Gross margin is unavailable.")
    elif gross_margin >= 0.40:
        supports.append(f"Gross margin appears strong at {format_percent(gross_margin)}.")
    elif gross_margin < 0.20:
        weakens.append(f"Gross margin appears thin at {format_percent(gross_margin)}.")

    if operating_margin is None:
        uncertainties.append("Operating margin is unavailable.")
    elif operating_margin > 0:
        supports.append(
            f"The company is operating-profitable with operating margin of "
            f"{format_percent(operating_margin)}."
        )
    else:
        weakens.append(
            f"The company has negative or zero operating margin "
            f"({format_percent(operating_margin)})."
        )

    if net_margin is None:
        uncertainties.append("Net margin is unavailable.")
    elif net_margin > 0:
        supports.append(f"Net margin is positive at {format_percent(net_margin)}.")
    else:
        weakens.append(f"Net margin is negative or zero at {format_percent(net_margin)}.")

    if free_cash_flow is None:
        uncertainties.append("Free cash flow could not be calculated.")
    elif free_cash_flow > 0:
        supports.append("Free cash flow is positive.")
    else:
        weakens.append("Free cash flow is negative or zero.")

    if free_cash_flow_margin is None:
        uncertainties.append("Free cash flow margin is unavailable.")
    elif free_cash_flow_margin >= 0.10:
        supports.append(
            f"Free cash flow margin appears healthy at "
            f"{format_percent(free_cash_flow_margin)}."
        )
    elif free_cash_flow_margin <= 0:
        weakens.append(
            f"Free cash flow margin is weak at {format_percent(free_cash_flow_margin)}."
        )

    if debt_to_equity is None:
        uncertainties.append("Debt-to-equity could not be calculated.")
    elif debt_to_equity <= 1:
        supports.append(f"Debt-to-equity appears manageable at {debt_to_equity:.2f}.")
    elif debt_to_equity >= 2:
        weakens.append(f"Debt-to-equity appears elevated at {debt_to_equity:.2f}.")

    return {
        "supports": supports,
        "weakens": weakens,
        "uncertainties": uncertainties,
    }


def classify_market_signals(market: dict[str, Any]) -> dict[str, list[str]]:
    """
    Create deterministic market and relative-performance signal labels.

    Args:
        market: Market snapshot dictionary.

    Returns:
        Dictionary with supports, weakens, and uncertainties lists.
    """

    supports: list[str] = []
    weakens: list[str] = []
    uncertainties: list[str] = []

    return_windows = market.get("return_windows", {})

    for window_name, values in return_windows.items():
        relative_return = values.get("relative_return") if isinstance(values, dict) else None
        stock_return = values.get("stock_return") if isinstance(values, dict) else None
        benchmark_return = (
            values.get("benchmark_return") if isinstance(values, dict) else None
        )

        readable_window = window_name.replace("_", " ")

        if relative_return is None:
            uncertainties.append(
                f"Relative return versus benchmark is unavailable for {readable_window}."
            )
        elif relative_return > 0:
            supports.append(
                f"The stock outperformed the benchmark over {readable_window} "
                f"by {format_percent(relative_return)}."
            )
        else:
            weakens.append(
                f"The stock underperformed the benchmark over {readable_window} "
                f"by {format_percent(abs(relative_return))}."
            )

        if stock_return is None or benchmark_return is None:
            continue

    annualized_volatility = market.get("annualized_volatility")
    benchmark_volatility = market.get("benchmark_annualized_volatility")

    if annualized_volatility is None:
        uncertainties.append("Annualized volatility is unavailable.")
    elif benchmark_volatility is not None and annualized_volatility > benchmark_volatility:
        weakens.append(
            "The stock has higher annualized volatility than the benchmark "
            f"({format_percent(annualized_volatility)} versus "
            f"{format_percent(benchmark_volatility)})."
        )
    elif benchmark_volatility is not None:
        supports.append(
            "The stock has annualized volatility at or below the benchmark "
            f"({format_percent(annualized_volatility)} versus "
            f"{format_percent(benchmark_volatility)})."
        )

    max_drawdown = market.get("max_drawdown")
    benchmark_max_drawdown = market.get("benchmark_max_drawdown")

    if max_drawdown is None:
        uncertainties.append("Max drawdown is unavailable.")
    elif benchmark_max_drawdown is not None and max_drawdown < benchmark_max_drawdown:
        weakens.append(
            "The stock experienced a deeper max drawdown than the benchmark "
            f"({format_percent(max_drawdown)} versus "
            f"{format_percent(benchmark_max_drawdown)})."
        )
    elif benchmark_max_drawdown is not None:
        supports.append(
            "The stock's max drawdown was less severe than or comparable to the benchmark "
            f"({format_percent(max_drawdown)} versus "
            f"{format_percent(benchmark_max_drawdown)})."
        )

    return {
        "supports": supports,
        "weakens": weakens,
        "uncertainties": uncertainties,
    }


def build_signal_summary(
    financial_snapshot: dict[str, Any],
    market_snapshot: dict[str, Any],
) -> dict[str, list[str]]:
    """
    Combine deterministic financial and market signals.

    Args:
        financial_snapshot: Financial evidence dictionary.
        market_snapshot: Market evidence dictionary.

    Returns:
        Combined signal summary.
    """

    financial_signals = classify_financial_signals(financial_snapshot)
    market_signals = classify_market_signals(market_snapshot)

    return {
        "supports": financial_signals["supports"] + market_signals["supports"],
        "weakens": financial_signals["weakens"] + market_signals["weakens"],
        "uncertainties": financial_signals["uncertainties"]
        + market_signals["uncertainties"],
    }


def build_base_evidence_package(
    symbol: str,
    analysis_type: str,
    cleaned_data: dict[str, pd.DataFrame],
    metrics: dict[str, Any],
    benchmark_symbol: str = "SPY",
) -> dict[str, Any]:
    """
    Build the base evidence package shared by all analysis types.

    Args:
        symbol: Stock ticker.
        analysis_type: Selected analysis type.
        cleaned_data: Cleaned DataFrame dictionary.
        metrics: Metric dictionary returned by calculate_all_metrics.
        benchmark_symbol: Benchmark ticker.

    Returns:
        Base evidence package dictionary.
    """

    financial_snapshot = get_financial_snapshot(metrics)
    market_snapshot = get_market_snapshot(metrics)
    signal_summary = build_signal_summary(financial_snapshot, market_snapshot)

    return {
        "symbol": symbol.upper(),
        "benchmark_symbol": benchmark_symbol.upper(),
        "analysis_type": analysis_type,
        "research_question": (
            f"Does the evidence support a credible setup for {symbol.upper()} "
            f"to outperform {benchmark_symbol.upper()}?"
        ),
        "company_overview": get_company_overview(cleaned_data),
        "financial_snapshot": financial_snapshot,
        "market_snapshot": market_snapshot,
        "recent_news": get_recent_news(cleaned_data, limit=5),
        "deterministic_signal_summary": signal_summary,
        "methodology_note": (
            "Evidence was assembled through deterministic routing based on the "
            "selected analysis type. No embeddings, vector database, or semantic "
            "retrieval were used."
        ),
    }


def build_overall_evidence(
    base_package: dict[str, Any],
) -> dict[str, Any]:
    """
    Build evidence for the overall outperformance thesis.

    Args:
        base_package: Shared evidence package.

    Returns:
        Prompt-specific evidence package.
    """

    return {
        **base_package,
        "evidence_focus": [
            "balanced financial quality",
            "benchmark-relative performance",
            "risk-adjusted context",
            "recent news context",
            "key uncertainties",
        ],
        "llm_instruction_focus": (
            "Weigh both supportive and weakening evidence. Explain whether the "
            "outperformance thesis is credible, mixed, or weak based on the "
            "provided evidence."
        ),
    }


def build_bull_case_evidence(
    base_package: dict[str, Any],
) -> dict[str, Any]:
    """
    Build evidence for the bull case.

    Args:
        base_package: Shared evidence package.

    Returns:
        Prompt-specific evidence package.
    """

    signals = base_package.get("deterministic_signal_summary", {})

    return {
        **base_package,
        "evidence_focus": [
            "evidence that supports an outperformance thesis",
            "positive financial quality signals",
            "positive relative-performance signals",
            "news items that may support investor interest",
        ],
        "primary_supporting_signals": signals.get("supports", []),
        "important_caveats": signals.get("weakens", [])[:5]
        + signals.get("uncertainties", [])[:5],
        "llm_instruction_focus": (
            "Construct the strongest reasonable bull case, but do not ignore "
            "material caveats or uncertainties."
        ),
    }


def build_bear_case_evidence(
    base_package: dict[str, Any],
) -> dict[str, Any]:
    """
    Build evidence for the bear case.

    Args:
        base_package: Shared evidence package.

    Returns:
        Prompt-specific evidence package.
    """

    signals = base_package.get("deterministic_signal_summary", {})

    return {
        **base_package,
        "evidence_focus": [
            "evidence that weakens an outperformance thesis",
            "financial quality concerns",
            "benchmark-relative underperformance",
            "volatility and drawdown risk",
            "uncertainties that limit confidence",
        ],
        "primary_weakening_signals": signals.get("weakens", []),
        "offsetting_supportive_signals": signals.get("supports", [])[:5],
        "unresolved_uncertainties": signals.get("uncertainties", []),
        "llm_instruction_focus": (
            "Construct the strongest reasonable bear case against outperformance, "
            "while acknowledging offsetting supportive evidence."
        ),
    }


def build_quality_evidence(
    base_package: dict[str, Any],
) -> dict[str, Any]:
    """
    Build evidence for financial quality assessment.

    Args:
        base_package: Shared evidence package.

    Returns:
        Prompt-specific evidence package.
    """

    return {
        **base_package,
        "evidence_focus": [
            "revenue growth",
            "margin structure",
            "cash generation",
            "balance sheet strength",
            "debt burden",
        ],
        "market_snapshot": {
            "included_for_context_only": True,
            "latest_stock_price": base_package.get("market_snapshot", {}).get(
                "latest_stock_price"
            ),
            "price_end_date": base_package.get("market_snapshot", {}).get(
                "price_end_date"
            ),
        },
        "llm_instruction_focus": (
            "Focus on business and financial quality rather than stock momentum. "
            "Assess whether the company's fundamentals strengthen or weaken the "
            "outperformance thesis."
        ),
    }


def build_improvement_evidence(
    base_package: dict[str, Any],
) -> dict[str, Any]:
    """
    Build evidence for the 'what would need to improve' analysis.

    Args:
        base_package: Shared evidence package.

    Returns:
        Prompt-specific evidence package.
    """

    signals = base_package.get("deterministic_signal_summary", {})

    return {
        **base_package,
        "evidence_focus": [
            "weak or missing evidence",
            "financial metrics that would need improvement",
            "market performance gaps versus benchmark",
            "risk metrics that would need to improve",
            "uncertainties requiring follow-up research",
        ],
        "current_weakening_signals": signals.get("weakens", []),
        "current_uncertainties": signals.get("uncertainties", []),
        "existing_supportive_signals": signals.get("supports", [])[:5],
        "llm_instruction_focus": (
            "Identify what would need to improve for the outperformance thesis "
            "to become more credible. Be specific and tie each improvement area "
            "to the evidence provided."
        ),
    }


def build_evidence_package(
    symbol: str,
    analysis_type: str,
    cleaned_data: dict[str, pd.DataFrame],
    metrics: dict[str, Any],
    benchmark_symbol: str = "SPY",
) -> dict[str, Any]:
    """
    Build a deterministic evidence package for a selected analysis type.

    Args:
        symbol: Stock ticker.
        analysis_type: Selected analysis type.
        cleaned_data: Cleaned DataFrame dictionary.
        metrics: Metric dictionary returned by calculate_all_metrics.
        benchmark_symbol: Benchmark ticker.

    Returns:
        Evidence package dictionary.

    Raises:
        ValueError: If analysis_type is unsupported.
    """

    if analysis_type not in SUPPORTED_ANALYSIS_TYPES:
        raise ValueError(
            f"Unsupported analysis_type: {analysis_type}. "
            f"Expected one of: {SUPPORTED_ANALYSIS_TYPES}"
        )

    base_package = build_base_evidence_package(
        symbol=symbol,
        analysis_type=analysis_type,
        cleaned_data=cleaned_data,
        metrics=metrics,
        benchmark_symbol=benchmark_symbol,
    )

    if analysis_type == ANALYSIS_OVERALL:
        evidence_package = build_overall_evidence(base_package)
    elif analysis_type == ANALYSIS_BULL:
        evidence_package = build_bull_case_evidence(base_package)
    elif analysis_type == ANALYSIS_BEAR:
        evidence_package = build_bear_case_evidence(base_package)
    elif analysis_type == ANALYSIS_QUALITY:
        evidence_package = build_quality_evidence(base_package)
    elif analysis_type == ANALYSIS_IMPROVEMENT:
        evidence_package = build_improvement_evidence(base_package)
    else:
        # Defensive fallback. The earlier validation should prevent this.
        raise ValueError(f"Unsupported analysis_type: {analysis_type}")

    return serialize_value(evidence_package)