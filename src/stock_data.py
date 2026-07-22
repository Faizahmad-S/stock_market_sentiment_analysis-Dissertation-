"""Historical price retrieval via `yfinance`.

Thin wrapper around ``yfinance`` that returns the columns actually used in
the sentiment/price correlation analysis (``Open`` and ``Close``), and
provides a helper to merge daily prices with a daily sentiment series.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Union

import pandas as pd
import yfinance as yf

DateLike = Union[str, date, datetime]


def fetch_prices(
    ticker: str,
    start: DateLike,
    end: DateLike,
    columns: tuple[str, ...] = ("Open", "Close"),
) -> pd.DataFrame:
    """Return historical prices for ``ticker`` between ``start`` and ``end``.

    The returned DataFrame is indexed by date (tz-naive) and contains only
    the columns listed in ``columns``.
    """
    data = yf.Ticker(ticker).history(start=str(start), end=str(end))
    if data.empty:
        return data

    # yfinance returns a tz-aware DatetimeIndex; strip the tz so it aligns
    # cleanly with the tz-naive dates coming from the headlines.
    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)

    return data[list(columns)].copy()


def daily_sentiment(
    df: pd.DataFrame,
    ticker: str,
    sentiment_column: str = "VADER_Sentiment",
) -> pd.DataFrame:
    """Return the mean daily sentiment for ``ticker``."""
    subset = df[df["Ticker"] == ticker]
    return (
        subset.groupby(subset["Date"].dt.normalize())[sentiment_column]
        .mean()
        .rename_axis("Date")
        .reset_index()
    )


def merge_sentiment_with_prices(
    sentiment_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join a daily sentiment series with a daily price series."""
    sentiment = sentiment_df.copy()
    sentiment["Date"] = pd.to_datetime(sentiment["Date"]).dt.normalize()

    prices = prices_df.copy()
    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices.reset_index().rename(columns={"index": "Date", "Date": "Date"})

    return sentiment.merge(prices, on="Date", how="left")
