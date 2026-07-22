"""Reusable plotting functions used throughout the analysis.

Every function accepts an optional ``ax`` so it can be composed inside a
larger figure, and an optional ``save_path`` so it can dump the resulting
figure to disk (used to regenerate the images under ``images/``). If neither
is provided, ``plt.show()`` is called at the end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import ensure_dir


def _finalize(fig: plt.Figure, save_path: Path | str | None) -> plt.Figure:
    if save_path is not None:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Sentiment distribution
# ---------------------------------------------------------------------------
def plot_sentiment_distribution(
    df: pd.DataFrame,
    score_column: str = "VADER_Sentiment",
    bins: int = 50,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Histogram of raw VADER sentiment scores (Figure 1 in the dissertation)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df[score_column].hist(bins=bins, color="blue", edgecolor="black", ax=ax)
    ax.set_title("Distribution of VADER Sentiment Scores")
    ax.set_xlabel("VADER Sentiment Score")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    return _finalize(fig, save_path)


def plot_sentiment_comparison(
    df: pd.DataFrame,
    score_column: str = "VADER_Sentiment",
    group_column: str = "Ticker",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Box plot comparing sentiment across tickers (Figure 2)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column=score_column, by=group_column, grid=False, ax=ax)
    ax.set_title("Sentiment Comparison Across Stocks")
    fig.suptitle("")  # drop the auto-generated suptitle
    ax.set_xlabel(group_column)
    ax.set_ylabel("VADER Sentiment Score")
    return _finalize(fig, save_path)


def plot_sentiment_categories(
    df: pd.DataFrame,
    category_column: str = "Sentiment_Category",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Bar chart of negative/neutral/positive counts (Figure 3)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    df[category_column].value_counts().plot(
        kind="bar", color=["red", "gray", "green"], ax=ax
    )
    ax.set_title("Distribution of Sentiment Categories")
    ax.set_xlabel("Sentiment Category")
    ax.set_ylabel("Count")
    ax.grid(True)
    return _finalize(fig, save_path)


def plot_articles_per_ticker(
    df: pd.DataFrame,
    ticker_column: str = "Ticker",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Bar chart of the number of scraped articles per ticker (Figure 4)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    df[ticker_column].value_counts().plot(kind="bar", color="skyblue", ax=ax)
    ax.set_title("Number of Articles per Ticker")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Number of Articles")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True)
    return _finalize(fig, save_path)


# ---------------------------------------------------------------------------
# Word cloud
# ---------------------------------------------------------------------------
def plot_wordcloud(
    df: pd.DataFrame,
    text_column: str = "Title",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Word cloud of scraped article titles (Figure 5)."""
    from wordcloud import WordCloud  # imported lazily; heavy dependency

    text = " ".join(df[text_column].dropna().astype(str).tolist())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud of Article Titles")
    return _finalize(fig, save_path)


# ---------------------------------------------------------------------------
# Sentiment over time
# ---------------------------------------------------------------------------
def plot_average_sentiment_per_day(
    df: pd.DataFrame,
    score_column: str = "VADER_Sentiment",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Bar chart of average sentiment per day (Figure 6)."""
    daily = df.groupby("Date")[score_column].mean()
    daily.index = daily.index.strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(12, 6))
    daily.plot(kind="bar", color="skyblue", ax=ax)
    ax.set_title("Average Sentiment Per Day")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average VADER Sentiment Score")
    ax.tick_params(axis="x", rotation=45)
    return _finalize(fig, save_path)


def plot_ticker_sentiment_trend(
    df: pd.DataFrame,
    ticker: str,
    score_column: str = "VADER_Sentiment",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Line chart of one ticker's mean daily sentiment (Figure 7)."""
    subset = df[df["Ticker"] == ticker]
    daily = subset.groupby("Date")[score_column].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily.index, daily.values, color="blue", label="Mean VADER Sentiment")
    ax.set_title(f"Mean VADER Sentiment Trend for {ticker} Per Day")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Sentiment Score")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True)
    ax.legend()
    return _finalize(fig, save_path)


def plot_sentiment_heatmap(
    df: pd.DataFrame,
    score_column: str = "VADER_Sentiment",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Heat map of sentiment × ticker × date (Figure 8)."""
    pivot = df.pivot_table(index="Date", columns="Ticker", values=score_column)
    pivot.index = pivot.index.strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, cmap="RdYlGn", linewidths=0.5, annot=False, ax=ax)
    ax.set_title("Sentiment Heatmap for Different Stocks Over Time")
    return _finalize(fig, save_path)


def plot_mean_sentiment_by_date_ticker(
    df: pd.DataFrame,
    score_column: str = "VADER_Sentiment",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Grouped bar chart of mean sentiment per date × ticker (Figure 9)."""
    grouped = df.groupby(["Date", "Ticker"])[score_column].mean().unstack()
    grouped.index = grouped.index.strftime("%Y-%m-%d")

    ax = grouped.plot(kind="bar", figsize=(12, 6), colormap="viridis")
    ax.set_title("Mean Sentiment per Date for Each Ticker")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean VADER Sentiment Score")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True)
    ax.legend(title="Ticker")
    fig = ax.get_figure()
    fig.tight_layout()
    return _finalize(fig, save_path)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: Iterable[str],
    title: str = "Confusion Matrix",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Heat map of a confusion matrix (Figures 11 & 13)."""
    labels = list(labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    return _finalize(fig, save_path)


# ---------------------------------------------------------------------------
# Stock prices
# ---------------------------------------------------------------------------
def plot_stock_prices(
    prices: pd.DataFrame,
    ticker: str,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot Open/Close prices for a ticker over the covered range (Figure 15)."""
    if "Open" not in prices.columns or "Close" not in prices.columns:
        raise ValueError("prices DataFrame must contain 'Open' and 'Close'")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prices.index, prices["Open"], marker="o", label="Opening", color="blue")
    ax.plot(prices.index, prices["Close"], marker="s", label="Closing", color="orange")

    start = prices.index.min().strftime("%Y-%m-%d")
    end = prices.index.max().strftime("%Y-%m-%d")
    ax.set_title(f"Stock Prices {ticker} ({start} to {end})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    return _finalize(fig, save_path)
