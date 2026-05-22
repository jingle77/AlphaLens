"""
Data cleaning utilities for AlphaLens.

This module converts raw Financial Modeling Prep API responses into clean,
analysis-ready pandas DataFrames.

Responsibilities:
- Convert raw records into DataFrames
- Parse date columns
- Convert numeric columns
- Sort financial statements newest-to-oldest
- Sort price history oldest-to-newest
- Standardize adjusted close prices into `close_for_returns`
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


COMMON_DATE_COLUMNS = (
    "date",
    "filingDate",
    "acceptedDate",
    "calendarYear",
    "period",
    "publishedDate",
)

PRICE_CLOSE_CANDIDATES = (
    "adjClose",
    "adj_close",
    "adjustedClose",
    "adjusted_close",
    "close",
)


def records_to_dataframe(records: list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """
    Convert raw FMP records into a pandas DataFrame.

    Args:
        records: FMP response payload. Usually a list of dictionaries, but
            occasionally a single dictionary.

    Returns:
        pandas DataFrame.

    Raises:
        ValueError: If records are empty or not DataFrame-compatible.
    """

    if records is None:
        raise ValueError("Cannot convert None records into a DataFrame.")

    if isinstance(records, list):
        if not records:
            raise ValueError("Cannot convert an empty list into a DataFrame.")
        return pd.DataFrame(records)

    if isinstance(records, dict):
        if not records:
            raise ValueError("Cannot convert an empty dictionary into a DataFrame.")
        return pd.DataFrame([records])

    raise TypeError(
        "records must be a list of dictionaries or a dictionary. "
        f"Received: {type(records)}"
    )


def convert_date_columns(
    df: pd.DataFrame,
    date_columns: Iterable[str] = COMMON_DATE_COLUMNS,
) -> pd.DataFrame:
    """
    Convert known date columns to pandas datetime when present.

    Some fields, such as fiscal period, may not be valid dates. Invalid parsing
    is coerced to NaT only for columns where conversion is appropriate.

    Args:
        df: Input DataFrame.
        date_columns: Candidate date columns to parse.

    Returns:
        Copy of DataFrame with parsed date columns.
    """

    cleaned = df.copy()

    for column in date_columns:
        if column not in cleaned.columns:
            continue

        # `period` is often values like FY, Q1, Q2 and should not be parsed.
        if column == "period":
            continue

        cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    return cleaned


def convert_numeric_columns(
    df: pd.DataFrame,
    exclude_columns: Iterable[str] = COMMON_DATE_COLUMNS,
) -> pd.DataFrame:
    """
    Convert numeric-looking object columns to numeric dtype when possible.

    Args:
        df: Input DataFrame.
        exclude_columns: Columns that should not be converted to numeric.

    Returns:
        Copy of DataFrame with numeric-looking columns converted.
    """

    cleaned = df.copy()
    excluded = set(exclude_columns)

    for column in cleaned.columns:
        if column in excluded:
            continue

        if cleaned[column].dtype == "object":
            converted = pd.to_numeric(cleaned[column], errors="ignore")
            cleaned[column] = converted

    return cleaned


def clean_company_profile(profile_records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Clean company profile records.

    Args:
        profile_records: Raw company profile response from FMP.

    Returns:
        Cleaned company profile DataFrame.
    """

    df = records_to_dataframe(profile_records)
    df = convert_date_columns(df)
    df = convert_numeric_columns(df)

    return df


def clean_financial_statement(
    statement_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Clean a financial statement DataFrame.

    Financial statements are sorted newest-to-oldest because many financial
    metrics compare the most recent period against prior periods.

    Args:
        statement_records: Raw income statement, balance sheet, or cash flow
            statement response from FMP.

    Returns:
        Cleaned financial statement DataFrame sorted newest-to-oldest.
    """

    df = records_to_dataframe(statement_records)
    df = convert_date_columns(df)
    df = convert_numeric_columns(df)
    df = sort_financial_statement(df)

    return df


def sort_financial_statement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort a financial statement newest-to-oldest.

    Args:
        df: Financial statement DataFrame.

    Returns:
        Sorted DataFrame.
    """

    cleaned = df.copy()

    if "date" in cleaned.columns:
        cleaned = cleaned.sort_values("date", ascending=False)

    cleaned = cleaned.reset_index(drop=True)

    return cleaned


def clean_price_history(price_records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Clean historical price data.

    Price history is sorted oldest-to-newest because return calculations use
    forward chronological order.

    This function also creates the canonical `close_for_returns` column.

    Args:
        price_records: Raw adjusted price history response from FMP.

    Returns:
        Cleaned price history DataFrame sorted oldest-to-newest.

    Raises:
        ValueError: If no usable close/adjusted close column is found.
    """

    df = records_to_dataframe(price_records)
    df = convert_date_columns(df)
    df = convert_numeric_columns(df)
    df = standardize_close_for_returns(df)
    df = sort_price_history(df)

    return df


def standardize_close_for_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a canonical `close_for_returns` column from adjusted close data.

    Candidate columns are checked in priority order. Adjusted close fields are
    preferred over regular close fields.

    Args:
        df: Price history DataFrame.

    Returns:
        Copy of DataFrame with `close_for_returns`.

    Raises:
        ValueError: If no valid close-like column is found.
    """

    cleaned = df.copy()

    selected_column = None

    for candidate in PRICE_CLOSE_CANDIDATES:
        if candidate in cleaned.columns:
            selected_column = candidate
            break

    if selected_column is None:
        raise ValueError(
            "Could not find a close column for return calculations. "
            f"Expected one of: {PRICE_CLOSE_CANDIDATES}"
        )

    cleaned["close_for_returns"] = pd.to_numeric(
        cleaned[selected_column],
        errors="coerce",
    )

    if cleaned["close_for_returns"].isna().all():
        raise ValueError(
            f"Column '{selected_column}' exists but could not be converted "
            "to numeric close_for_returns values."
        )

    return cleaned


def sort_price_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort price history oldest-to-newest.

    Args:
        df: Price history DataFrame.

    Returns:
        Sorted DataFrame.

    Raises:
        ValueError: If the DataFrame does not contain a date column.
    """

    cleaned = df.copy()

    if "date" not in cleaned.columns:
        raise ValueError("Price history must contain a 'date' column.")

    cleaned = cleaned.dropna(subset=["date", "close_for_returns"])
    cleaned = cleaned.sort_values("date", ascending=True)
    cleaned = cleaned.reset_index(drop=True)

    return cleaned


def clean_news(news_records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Clean stock news records.

    News is allowed to be empty. In that case, this returns an empty DataFrame
    instead of raising an error.

    Args:
        news_records: Raw stock news response from FMP.

    Returns:
        Cleaned news DataFrame.
    """

    if not news_records:
        return pd.DataFrame()

    df = records_to_dataframe(news_records)
    df = convert_date_columns(df)
    df = convert_numeric_columns(df)

    if "publishedDate" in df.columns:
        df = df.sort_values("publishedDate", ascending=False).reset_index(drop=True)

    return df


def clean_research_data(raw_data: Any) -> dict[str, pd.DataFrame]:
    """
    Clean the full raw ResearchData object returned by data_fetcher.py.

    Args:
        raw_data: ResearchData object from src.data_fetcher.fetch_research_data.

    Returns:
        Dictionary of cleaned DataFrames.
    """

    return {
        "company_profile": clean_company_profile(raw_data.company_profile),
        "income_statement": clean_financial_statement(raw_data.income_statement),
        "balance_sheet": clean_financial_statement(raw_data.balance_sheet),
        "cash_flow": clean_financial_statement(raw_data.cash_flow),
        "price_history": clean_price_history(raw_data.price_history),
        "benchmark_price_history": clean_price_history(raw_data.benchmark_price_history),
        "stock_news": clean_news(raw_data.stock_news),
    }