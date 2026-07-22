"""Scrape financial news headlines from Finviz.

For each ticker in :data:`~src.utils.DEFAULT_TICKERS` (or a supplied list),
this module downloads the news table shown on ``finviz.com/quote.ashx`` and
parses each row into ``(ticker, date, time, title)`` tuples.

The scraper is designed to be run once per day (see the GitHub Actions
workflow at ``.github/workflows/scrape.yml``). Each run writes a timestamped
CSV to ``data/raw/`` so historical files are never overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup

from .utils import DEFAULT_TICKERS, RAW_DATA_DIR, ensure_dir, timestamp

FINVIZ_URL = "https://finviz.com/quote.ashx?t="
USER_AGENT = "Mozilla/5.0"


def fetch_news_table(ticker: str) -> BeautifulSoup | None:
    """Download the Finviz news table for ``ticker``.

    Parameters
    ----------
    ticker:
        Stock ticker symbol, e.g. ``"AAPL"``.

    Returns
    -------
    BeautifulSoup element or None
        The ``<table id="news-table">`` element, or ``None`` if the request
        failed.
    """
    url = FINVIZ_URL + ticker
    request = Request(url=url, headers={"user-agent": USER_AGENT})
    try:
        response = urlopen(request)
    except HTTPError as exc:
        print(f"Error fetching data for {ticker}: {exc}")
        return None

    html = BeautifulSoup(response, "html.parser")
    return html.find(id="news-table")


def parse_news_table(ticker: str, news_table) -> list[list[str]]:
    """Parse a Finviz news table into rows.

    Each Finviz row shows either a full timestamp ("Sep-25-24 07:15AM") or
    just a time when the article was published earlier the same day. We
    forward-fill the date so every row is fully qualified.
    """
    if news_table is None:
        return []

    parsed: list[list[str]] = []
    current_date = ""

    for row in news_table.findAll("tr"):
        if not row.a:
            continue
        title = row.a.text.strip()
        date_data = row.td.text.strip().split(" ")

        if len(date_data) == 1:
            time_ = date_data[0]
            date = current_date
        else:
            date, time_ = date_data[0], date_data[1]
            current_date = date

        parsed.append([ticker, date, time_, title])

    return parsed


def scrape(tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Scrape news for a list of tickers and return a DataFrame."""
    tickers = list(tickers) if tickers else DEFAULT_TICKERS
    rows: list[list[str]] = []
    for ticker in tickers:
        table = fetch_news_table(ticker)
        rows.extend(parse_news_table(ticker, table))
    return pd.DataFrame(rows, columns=["Ticker", "Date", "Time", "Title"])


def save_dataframe(df: pd.DataFrame, out_dir: Path | str = RAW_DATA_DIR) -> Path:
    """Persist ``df`` to ``out_dir`` with a timestamped filename."""
    out_dir = ensure_dir(out_dir)
    ts = timestamp()
    out_path = Path(out_dir) / f"news_data_{ts}.csv"
    df.to_csv(out_path, index=False)

    # Also write a timestamp.txt so the GitHub Actions workflow can pick it up
    (Path(out_dir) / "timestamp.txt").write_text(ts)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to scrape (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(RAW_DATA_DIR),
        help="Directory to write the CSV to (default: %(default)s)",
    )
    args = parser.parse_args()

    df = scrape(args.tickers)
    if df.empty:
        print("No headlines scraped.")
        return

    path = save_dataframe(df, args.out_dir)
    print(f"Wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
