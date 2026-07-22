"""Shared constants and small helpers used across the project."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Tickers scraped from Finviz. Extend this list to cover additional stocks.
# ---------------------------------------------------------------------------
DEFAULT_TICKERS: list[str] = ["AMZN", "GOOG", "AAPL", "WMT"]

# ---------------------------------------------------------------------------
# Sentiment thresholds. The compound / polarity score is bucketed into
# {negative, neutral, positive} using these cut-offs.
# ---------------------------------------------------------------------------
NEGATIVE_THRESHOLD: float = -0.2
POSITIVE_THRESHOLD: float = 0.2

# ---------------------------------------------------------------------------
# Repository paths (resolved relative to this file so scripts run from any
# working directory).
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
IMAGES_DIR: Path = PROJECT_ROOT / "images"
RESULTS_DIR: Path = PROJECT_ROOT / "results"


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) if it does not exist and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """Return the current local timestamp formatted as ``fmt``."""
    return datetime.now().strftime(fmt)
