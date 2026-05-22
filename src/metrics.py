"""
Deterministic financial and market metric calculations for AlphaLens.

This module receives cleaned pandas DataFrames from src/data_cleaner.py and
calculates the structured metrics used by the evidence and LLM synthesis layers.

The LLM should not be responsible for calculating these metrics. Python should
produce the numbers; the LLM should interpret them.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252

RETURN_WINDOWS = {
    "1_month": 21,
    "3_month": 63,
    "6_month": 126,
    "1_year": 252,
}


REVENUE_COLUMNS = ("revenue", "totalRevenue")
GROSS_PROFIT_COLUMNS = ("grossProfit",)
OPERATING_INCOME_COLUMNS = ("operatingIncome",)
NET_INCOME_COLUMNS = ("netIncome",)

OPERATING_CASH_FLOW_COLUMNS = (
    "operatingCashFlow",
    "netCashProvidedByOperatingActivities",
)

CAPITAL_EXPENDITURE_COLUMNS = (
    "capitalExpenditure",
    "capitalExpenditures",
)

FREE_CASH_FLOW_COLUMNS = (
    "freeCashFlow",
)

CASH_COLUMNS = (
    "cashAndCashEquivalents",
    "cashAndShortTermInvestments",
    "cashAndCashEquivalentsAtCarryingValue",
)

TOTAL_DEBT_COLUMNS = (
    "totalDebt",
)

SHORT_TERM_DEBT_COLUMNS = (
    "shortTermDebt",
    "shortTermBorrowings",
)

LONG_TERM_DEBT_COLUMNS = (
    "longTermDebt",
    "longTermDebtNoncurrent",
)

EQUITY_COLUMNS = (
    "totalStockholdersEquity",
    "totalEquity",
    "stockholdersEquity",
)


def is_valid_number(value: Any) -> bool:
    """
    Check whether a value is a finite numeric value.

    Args:
        value: Value to check.

    Returns:
        True if value is numeric, non-null, and finite.
    """

    if value is None:
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return not math.isnan(numeric_value) and math.isfinite(numeric_value)


def to_float_or_none(value: Any) -> float | None:
    """
    Convert a value to float if possible.

    Args:
        value: Value to convert.

    Returns:
        Float value or None.
    """

    if not is_valid_number(value):
        return None

    return float(value)


def safe_divide(numerator: Any, denominator: Any) -> float | None:
    """
    Safely divide two values.

    Args:
        numerator: Numerator.
        denominator: Denominator.

    Returns:
        numerator / denominator, or None if invalid.
    """

    numerator = to_float_or_none(numerator)
    denominator = to_float_or_none(denominator)

    if numerator is None or denominator is None:
        return None

    if denominator == 0:
        return None

    return numerator / denominator


def percentage_change(current_value: Any, previous_value: Any) -> float | None:
    """
    Calculate percentage change from previous value to current value.

    Args:
        current_value: Current value.
        previous_value: Previous value.

    Returns:
        Percentage change as a decimal, or None if invalid.
        Example: 0.10 means 10%.
    """

    current_value = to_float_or_none(current_value)
    previous_value = to_float_or_none(previous_value)

    if current_value is None or previous_value is None:
        return None

    if previous_value == 0:
        return None

    return (current_value / previous_value) - 1


def get_first_available_value(
    row: pd.Series,
    candidate_columns: tuple[str, ...],
) -> float | None:
    """
    Return the first available numeric value from candidate columns.

    Args:
        row: DataFrame row.
        candidate_columns: Ordered list of candidate columns.

    Returns:
        First available numeric value or None.
    """

    for column in candidate_columns:
        if column in row.index:
            value = to_float_or_none(row[column])
            if value is not None:
                return value

    return None


def get_latest_row(df: pd.DataFrame) -> pd.Series | None:
    """
    Return the latest row from a cleaned financial statement.

    Financial statements should already be sorted newest-to-oldest by
    src/data_cleaner.py.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Latest row or None.
    """

    if df is None or df.empty:
        return None

    return df.iloc[0]


def get_prior_row(df: pd.DataFrame) -> pd.Series | None:
    """
    Return the prior row from a cleaned financial statement.

    Args:
        df: Cleaned DataFrame sorted newest-to-oldest.

    Returns:
        Prior row or None.
    """

    if df is None or len(df) < 2:
        return None

    return df.iloc[1]


def calculate_free_cash_flow(cash_flow_row: pd.Series | None) -> float | None:
    """
    Calculate free cash flow.

    Prefer FMP's freeCashFlow field when available. If unavailable, calculate
    from operating cash flow and capital expenditure.

    FMP capital expenditure is often reported as a negative number. In that
    common case, free cash flow = operating cash flow + capital expenditure.

    Args:
        cash_flow_row: Latest cash flow statement row.

    Returns:
        Free cash flow or None.
    """

    if cash_flow_row is None:
        return None

    reported_fcf = get_first_available_value(cash_flow_row, FREE_CASH_FLOW_COLUMNS)

    if reported_fcf is not None:
        return reported_fcf

    operating_cash_flow = get_first_available_value(
        cash_flow_row,
        OPERATING_CASH_FLOW_COLUMNS,
    )
    capital_expenditure = get_first_available_value(
        cash_flow_row,
        CAPITAL_EXPENDITURE_COLUMNS,
    )

    if operating_cash_flow is None or capital_expenditure is None:
        return None

    if capital_expenditure < 0:
        return operating_cash_flow + capital_expenditure

    return operating_cash_flow - capital_expenditure


def calculate_total_debt(balance_sheet_row: pd.Series | None) -> float | None:
    """
    Calculate total debt.

    Prefer FMP's totalDebt field. If unavailable, try to estimate it using
    short-term debt plus long-term debt.

    Args:
        balance_sheet_row: Latest balance sheet row.

    Returns:
        Total debt or None.
    """

    if balance_sheet_row is None:
        return None

    total_debt = get_first_available_value(balance_sheet_row, TOTAL_DEBT_COLUMNS)

    if total_debt is not None:
        return total_debt

    short_term_debt = get_first_available_value(
        balance_sheet_row,
        SHORT_TERM_DEBT_COLUMNS,
    )
    long_term_debt = get_first_available_value(
        balance_sheet_row,
        LONG_TERM_DEBT_COLUMNS,
    )

    if short_term_debt is None and long_term_debt is None:
        return None

    return (short_term_debt or 0.0) + (long_term_debt or 0.0)


def calculate_financial_metrics(
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate deterministic financial metrics.

    Args:
        income_statement: Cleaned income statement DataFrame.
        balance_sheet: Cleaned balance sheet DataFrame.
        cash_flow: Cleaned cash flow statement DataFrame.

    Returns:
        Dictionary of financial metrics.
    """

    latest_income = get_latest_row(income_statement)
    prior_income = get_prior_row(income_statement)
    latest_balance = get_latest_row(balance_sheet)
    latest_cash_flow = get_latest_row(cash_flow)

    latest_fiscal_date = None
    if latest_income is not None and "date" in latest_income.index:
        latest_fiscal_date = latest_income["date"]

    revenue = (
        get_first_available_value(latest_income, REVENUE_COLUMNS)
        if latest_income is not None
        else None
    )

    prior_revenue = (
        get_first_available_value(prior_income, REVENUE_COLUMNS)
        if prior_income is not None
        else None
    )

    gross_profit = (
        get_first_available_value(latest_income, GROSS_PROFIT_COLUMNS)
        if latest_income is not None
        else None
    )

    operating_income = (
        get_first_available_value(latest_income, OPERATING_INCOME_COLUMNS)
        if latest_income is not None
        else None
    )

    net_income = (
        get_first_available_value(latest_income, NET_INCOME_COLUMNS)
        if latest_income is not None
        else None
    )

    operating_cash_flow = (
        get_first_available_value(latest_cash_flow, OPERATING_CASH_FLOW_COLUMNS)
        if latest_cash_flow is not None
        else None
    )

    free_cash_flow = calculate_free_cash_flow(latest_cash_flow)

    cash_and_equivalents = (
        get_first_available_value(latest_balance, CASH_COLUMNS)
        if latest_balance is not None
        else None
    )

    total_debt = calculate_total_debt(latest_balance)

    total_equity = (
        get_first_available_value(latest_balance, EQUITY_COLUMNS)
        if latest_balance is not None
        else None
    )

    return {
        "latest_fiscal_date": latest_fiscal_date,
        "revenue": revenue,
        "prior_revenue": prior_revenue,
        "revenue_growth": percentage_change(revenue, prior_revenue),
        "gross_profit": gross_profit,
        "gross_margin": safe_divide(gross_profit, revenue),
        "operating_income": operating_income,
        "operating_margin": safe_divide(operating_income, revenue),
        "net_income": net_income,
        "net_margin": safe_divide(net_income, revenue),
        "operating_cash_flow": operating_cash_flow,
        "free_cash_flow": free_cash_flow,
        "free_cash_flow_margin": safe_divide(free_cash_flow, revenue),
        "cash_and_equivalents": cash_and_equivalents,
        "total_debt": total_debt,
        "total_equity": total_equity,
        "debt_to_equity": safe_divide(total_debt, total_equity),
    }


def validate_price_dataframe(price_history: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and prepare a price DataFrame for return calculations.

    Args:
        price_history: Cleaned price history DataFrame.

    Returns:
        Price DataFrame sorted oldest-to-newest.

    Raises:
        ValueError: If required columns are missing.
    """

    if price_history is None or price_history.empty:
        raise ValueError("price_history cannot be empty.")

    required_columns = {"date", "close_for_returns"}
    missing_columns = required_columns - set(price_history.columns)

    if missing_columns:
        raise ValueError(
            f"price_history is missing required columns: {sorted(missing_columns)}"
        )

    df = price_history.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close_for_returns"] = pd.to_numeric(
        df["close_for_returns"],
        errors="coerce",
    )

    df = df.dropna(subset=["date", "close_for_returns"])
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    if df.empty:
        raise ValueError("price_history has no valid price rows after cleaning.")

    return df


def calculate_period_return(
    price_history: pd.DataFrame,
    trading_days: int,
) -> float | None:
    """
    Calculate return over a fixed trading-day lookback window.

    Args:
        price_history: Cleaned price history DataFrame sorted oldest-to-newest.
        trading_days: Lookback window in trading days.

    Returns:
        Return as a decimal, or None if insufficient data.
    """

    df = validate_price_dataframe(price_history)

    if trading_days <= 0:
        raise ValueError("trading_days must be greater than 0.")

    if len(df) <= trading_days:
        return None

    start_price = df["close_for_returns"].iloc[-trading_days - 1]
    end_price = df["close_for_returns"].iloc[-1]

    return percentage_change(end_price, start_price)


def calculate_return_windows(
    stock_price_history: pd.DataFrame,
    benchmark_price_history: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    """
    Calculate stock, benchmark, and relative returns over standard windows.

    Args:
        stock_price_history: Cleaned stock price history.
        benchmark_price_history: Cleaned benchmark price history.

    Returns:
        Nested dictionary of return metrics.
    """

    results: dict[str, dict[str, float | None]] = {}

    for window_name, trading_days in RETURN_WINDOWS.items():
        stock_return = calculate_period_return(stock_price_history, trading_days)
        benchmark_return = calculate_period_return(
            benchmark_price_history,
            trading_days,
        )

        relative_return = None
        if stock_return is not None and benchmark_return is not None:
            relative_return = stock_return - benchmark_return

        results[window_name] = {
            "trading_days": trading_days,
            "stock_return": stock_return,
            "benchmark_return": benchmark_return,
            "relative_return": relative_return,
        }

    return results


def calculate_annualized_volatility(price_history: pd.DataFrame) -> float | None:
    """
    Calculate annualized volatility from daily returns.

    Args:
        price_history: Cleaned price history DataFrame.

    Returns:
        Annualized volatility as a decimal, or None if insufficient data.
    """

    df = validate_price_dataframe(price_history)

    daily_returns = df["close_for_returns"].pct_change().dropna()

    if len(daily_returns) < 2:
        return None

    daily_volatility = daily_returns.std()

    if not is_valid_number(daily_volatility):
        return None

    return float(daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR))


def calculate_max_drawdown(price_history: pd.DataFrame) -> float | None:
    """
    Calculate max drawdown from close_for_returns.

    Max drawdown is the largest percentage decline from a prior peak.

    Args:
        price_history: Cleaned price history DataFrame.

    Returns:
        Max drawdown as a negative decimal, or None if insufficient data.
        Example: -0.25 means a 25% drawdown.
    """

    df = validate_price_dataframe(price_history)

    if len(df) < 2:
        return None

    prices = df["close_for_returns"]
    running_peak = prices.cummax()
    drawdowns = (prices / running_peak) - 1

    max_drawdown = drawdowns.min()

    if not is_valid_number(max_drawdown):
        return None

    return float(max_drawdown)


def calculate_market_metrics(
    stock_price_history: pd.DataFrame,
    benchmark_price_history: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate deterministic market metrics versus benchmark.

    Args:
        stock_price_history: Cleaned stock price history.
        benchmark_price_history: Cleaned benchmark price history.

    Returns:
        Dictionary of market metrics.
    """

    stock_prices = validate_price_dataframe(stock_price_history)
    benchmark_prices = validate_price_dataframe(benchmark_price_history)

    return {
        "price_start_date": stock_prices["date"].iloc[0],
        "price_end_date": stock_prices["date"].iloc[-1],
        "latest_stock_price": to_float_or_none(
            stock_prices["close_for_returns"].iloc[-1]
        ),
        "latest_benchmark_price": to_float_or_none(
            benchmark_prices["close_for_returns"].iloc[-1]
        ),
        "return_windows": calculate_return_windows(stock_prices, benchmark_prices),
        "annualized_volatility": calculate_annualized_volatility(stock_prices),
        "benchmark_annualized_volatility": calculate_annualized_volatility(
            benchmark_prices
        ),
        "max_drawdown": calculate_max_drawdown(stock_prices),
        "benchmark_max_drawdown": calculate_max_drawdown(benchmark_prices),
    }


def calculate_all_metrics(cleaned_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Calculate the full AlphaLens metric package.

    Args:
        cleaned_data: Dictionary returned by src.data_cleaner.clean_research_data.

    Returns:
        Dictionary containing financial and market metrics.
    """

    required_keys = {
        "income_statement",
        "balance_sheet",
        "cash_flow",
        "price_history",
        "benchmark_price_history",
    }

    missing_keys = required_keys - set(cleaned_data)

    if missing_keys:
        raise ValueError(f"cleaned_data is missing required keys: {sorted(missing_keys)}")

    financial_metrics = calculate_financial_metrics(
        income_statement=cleaned_data["income_statement"],
        balance_sheet=cleaned_data["balance_sheet"],
        cash_flow=cleaned_data["cash_flow"],
    )

    market_metrics = calculate_market_metrics(
        stock_price_history=cleaned_data["price_history"],
        benchmark_price_history=cleaned_data["benchmark_price_history"],
    )

    return {
        "financial": financial_metrics,
        "market": market_metrics,
    }