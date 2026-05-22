"""
Streamlit frontend for AlphaLens.

AlphaLens is an equity research assistant that evaluates the evidence-weighted
case that a selected stock could outperform the S&P 500.

This file should remain UI-focused. Core API, metric, evidence, prompt, and LLM
logic lives in the src/ package.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config import settings
from src.prompts import get_analysis_options
from src.research_assistant import (
    EquityAnalysisResult,
    ResearchAssistantError,
    generate_equity_analysis,
)


st.set_page_config(
    page_title="AlphaLens",
    page_icon="🔎",
    layout="wide",
)


def format_percent(value: Any) -> str:
    """
    Format a decimal value as a percentage.

    Example:
        0.123 -> "12.3%"
    """

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def format_number(value: Any) -> str:
    """
    Format a numeric value compactly.
    """

    if value is None:
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    abs_value = abs(value)

    if abs_value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"

    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"

    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"

    return f"${value:,.0f}"


def format_decimal(value: Any) -> str:
    """
    Format a plain decimal value.
    """

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def get_return_metric(
    result: EquityAnalysisResult,
    window: str,
    field: str,
) -> Any:
    """
    Safely extract a return-window metric from an AlphaLens result.
    """

    return (
        result.market_metrics
        .get("return_windows", {})
        .get(window, {})
        .get(field)
    )


@st.cache_data(show_spinner=False, ttl=900)
def run_analysis_cached(
    symbol: str,
    analysis_type: str,
    benchmark_symbol: str,
    statement_limit: int,
    news_limit: int,
    price_history_days: int,
) -> EquityAnalysisResult:
    """
    Run the AlphaLens backend pipeline with short-term Streamlit caching.

    The cache prevents duplicate FMP/OpenAI calls when Streamlit reruns after
    minor UI interactions.
    """

    return generate_equity_analysis(
        symbol=symbol,
        analysis_type=analysis_type,
        benchmark_symbol=benchmark_symbol,
        statement_limit=statement_limit,
        news_limit=news_limit,
        price_history_days=price_history_days,
    )


def render_header() -> None:
    """
    Render the main page header.
    """

    st.title("AlphaLens")
    st.caption(
        "Evidence-weighted equity research assistant for evaluating whether a "
        "selected stock has a credible setup to outperform the S&P 500."
    )

    st.info(
        "AlphaLens is a research tool, not a financial advisor. It does not make "
        "buy, sell, hold, or price-target recommendations."
    )


def render_sidebar() -> dict[str, Any]:
    """
    Render sidebar controls.

    Returns:
        Dictionary of user-selected settings.
    """

    st.sidebar.header("Analysis settings")

    ticker_mode = st.sidebar.radio(
        "Ticker input mode",
        options=["Select from list", "Enter manually"],
        horizontal=False,
    )

    if ticker_mode == "Select from list":
        selected_symbol = st.sidebar.selectbox(
            "Stock ticker",
            options=list(settings.default_ticker_list),
            index=0,
        )
    else:
        selected_symbol = st.sidebar.text_input(
            "Stock ticker",
            value="AAPL",
            max_chars=12,
        )

    analysis_type = st.sidebar.selectbox(
        "Analysis angle",
        options=get_analysis_options(),
        index=0,
    )

    with st.sidebar.expander("Advanced settings", expanded=False):
        benchmark_symbol = st.text_input(
            "Benchmark symbol",
            value=settings.default_benchmark_symbol,
            max_chars=12,
        )

        statement_limit = st.slider(
            "Financial statement periods",
            min_value=2,
            max_value=10,
            value=settings.default_statement_limit,
            step=1,
        )

        news_limit = st.slider(
            "Recent news articles",
            min_value=0,
            max_value=20,
            value=settings.default_news_limit,
            step=1,
        )

        price_history_days = st.slider(
            "Price history lookback days",
            min_value=260,
            max_value=1_500,
            value=settings.default_price_history_days,
            step=20,
        )

    generate_clicked = st.sidebar.button(
        "Generate analysis",
        type="primary",
        use_container_width=True,
    )

    return {
        "symbol": selected_symbol.strip().upper(),
        "analysis_type": analysis_type,
        "benchmark_symbol": benchmark_symbol.strip().upper(),
        "statement_limit": statement_limit,
        "news_limit": news_limit,
        "price_history_days": price_history_days,
        "generate_clicked": generate_clicked,
    }


def render_key_metrics(result: EquityAnalysisResult) -> None:
    """
    Render key metric cards.
    """

    financial = result.financial_metrics

    one_year_stock_return = get_return_metric(result, "1_year", "stock_return")
    one_year_benchmark_return = get_return_metric(result, "1_year", "benchmark_return")
    one_year_relative_return = get_return_metric(result, "1_year", "relative_return")

    st.subheader("Key metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="Revenue growth",
        value=format_percent(financial.get("revenue_growth")),
    )

    col2.metric(
        label="Operating margin",
        value=format_percent(financial.get("operating_margin")),
    )

    col3.metric(
        label="FCF margin",
        value=format_percent(financial.get("free_cash_flow_margin")),
    )

    col4.metric(
        label="Debt / equity",
        value=format_decimal(financial.get("debt_to_equity")),
    )

    col5, col6, col7, col8 = st.columns(4)

    col5.metric(
        label="1Y stock return",
        value=format_percent(one_year_stock_return),
    )

    col6.metric(
        label="1Y benchmark return",
        value=format_percent(one_year_benchmark_return),
    )

    col7.metric(
        label="1Y relative return",
        value=format_percent(one_year_relative_return),
    )

    col8.metric(
        label="Annualized volatility",
        value=format_percent(result.market_metrics.get("annualized_volatility")),
    )


def render_metric_details(result: EquityAnalysisResult) -> None:
    """
    Render expandable financial and market metric details.
    """

    with st.expander("Financial metrics", expanded=False):
        financial = result.financial_metrics

        metric_rows = {
            "Latest fiscal date": financial.get("latest_fiscal_date"),
            "Revenue": format_number(financial.get("revenue")),
            "Prior revenue": format_number(financial.get("prior_revenue")),
            "Revenue growth": format_percent(financial.get("revenue_growth")),
            "Gross margin": format_percent(financial.get("gross_margin")),
            "Operating margin": format_percent(financial.get("operating_margin")),
            "Net margin": format_percent(financial.get("net_margin")),
            "Operating cash flow": format_number(financial.get("operating_cash_flow")),
            "Free cash flow": format_number(financial.get("free_cash_flow")),
            "Free cash flow margin": format_percent(
                financial.get("free_cash_flow_margin")
            ),
            "Cash and equivalents": format_number(
                financial.get("cash_and_equivalents")
            ),
            "Total debt": format_number(financial.get("total_debt")),
            "Total equity": format_number(financial.get("total_equity")),
            "Debt-to-equity": format_decimal(financial.get("debt_to_equity")),
        }

        st.table(metric_rows)

    with st.expander("Market metrics", expanded=False):
        market = result.market_metrics

        st.write("**Price context**")
        st.table(
            {
                "Price start date": market.get("price_start_date"),
                "Price end date": market.get("price_end_date"),
                "Latest stock price": format_number(market.get("latest_stock_price")),
                "Latest benchmark price": format_number(
                    market.get("latest_benchmark_price")
                ),
                "Annualized volatility": format_percent(
                    market.get("annualized_volatility")
                ),
                "Benchmark annualized volatility": format_percent(
                    market.get("benchmark_annualized_volatility")
                ),
                "Max drawdown": format_percent(market.get("max_drawdown")),
                "Benchmark max drawdown": format_percent(
                    market.get("benchmark_max_drawdown")
                ),
            }
        )

        st.write("**Return windows**")

        return_rows = []

        for window, values in market.get("return_windows", {}).items():
            return_rows.append(
                {
                    "Window": window.replace("_", " ").title(),
                    "Trading days": values.get("trading_days"),
                    "Stock return": format_percent(values.get("stock_return")),
                    "Benchmark return": format_percent(values.get("benchmark_return")),
                    "Relative return": format_percent(values.get("relative_return")),
                }
            )

        st.dataframe(return_rows, use_container_width=True)


def render_evidence_details(result: EquityAnalysisResult) -> None:
    """
    Render expandable evidence and diagnostic details.
    """

    with st.expander("Deterministic evidence package", expanded=False):
        st.caption(
            "This evidence was assembled by deterministic evidence routing. "
            "AlphaLens v1 does not use embeddings, vector search, or semantic RAG."
        )
        st.json(result.evidence_package)

    with st.expander("Data diagnostics", expanded=False):
        st.write("Cleaned dataset shapes:")
        st.table(result.cleaned_data_shapes)


def render_result(result: EquityAnalysisResult) -> None:
    """
    Render the completed AlphaLens result.
    """

    st.header(f"{result.symbol} vs. {result.benchmark_symbol}")
    st.subheader(result.analysis_type)

    st.markdown(result.analysis_text)

    st.divider()

    render_key_metrics(result)
    render_metric_details(result)
    render_evidence_details(result)


def main() -> None:
    """
    Main Streamlit app entry point.
    """

    render_header()

    selections = render_sidebar()

    st.write(
        "Choose a ticker and analysis angle, then generate an evidence-weighted "
        "research note."
    )

    if not selections["generate_clicked"]:
        st.stop()

    if not selections["symbol"]:
        st.error("Please enter a valid stock ticker.")
        st.stop()

    with st.spinner("Fetching data, calculating metrics, routing evidence, and generating analysis..."):
        try:
            result = run_analysis_cached(
                symbol=selections["symbol"],
                analysis_type=selections["analysis_type"],
                benchmark_symbol=selections["benchmark_symbol"],
                statement_limit=selections["statement_limit"],
                news_limit=selections["news_limit"],
                price_history_days=selections["price_history_days"],
            )
        except ResearchAssistantError as exc:
            st.error("AlphaLens could not generate the analysis.")
            st.exception(exc)
            st.stop()
        except Exception as exc:
            st.error("Unexpected application error.")
            st.exception(exc)
            st.stop()

    render_result(result)


if __name__ == "__main__":
    main()