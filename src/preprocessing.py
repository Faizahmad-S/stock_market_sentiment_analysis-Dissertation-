"""Merge and clean the raw scraped headline CSVs.

The daily scraper writes one CSV per run into ``data/raw/``. This module
walks that directory (or reads a ZIP archive downloaded from GitHub), merges
every CSV into a single DataFrame, drops invalid or duplicate rows and
returns a chronologically sorted result ready for sentiment analysis.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import RAW_DATA_DIR

EXPECTED_COLUMNS = ["Ticker", "Date", "Time", "Title"]


def load_from_directory(directory: Path | str = RAW_DATA_DIR) -> pd.DataFrame:
    """Read every ``.csv`` file in ``directory`` and concatenate them."""
    directory = Path(directory)
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    frames = [pd.read_csv(f) for f in csv_files]
    return pd.concat(frames, ignore_index=True)


def load_from_zip(zip_path: Path | str) -> pd.DataFrame:
    """Read every CSV inside ``zip_path`` and concatenate them.

    Useful when the GitHub repository has been downloaded as a ZIP archive
    (as in the original Colab workflow).
    """
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            if not name.endswith(".csv"):
                continue
            with archive.open(name) as fh:
                frames.append(pd.read_csv(fh))
    if not frames:
        raise ValueError(f"No CSV files inside {zip_path}")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a merged headline DataFrame.

    Steps:
      * drop rows whose ``Date`` column literally says ``"today"``
      * drop exact duplicates
      * parse ``Date`` into a proper ``datetime64``
      * sort by ``(Ticker, Date)`` ascending

    A new DataFrame is returned; the input is not modified.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    out = df.copy()
    out = out[~out["Date"].astype(str).str.contains("today", case=False, na=False)]
    out = out.drop_duplicates()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])
    out = out.sort_values(by=["Ticker", "Date"], ascending=[True, True])
    return out.reset_index(drop=True)


def preprocess(source: Path | str | Iterable[Path] | None = None) -> pd.DataFrame:
    """One-shot helper: load from ``source`` and return a cleaned DataFrame.

    ``source`` can be a directory (default: ``data/raw``), a ``.zip`` file,
    or ``None`` to fall back to the default raw directory.
    """
    if source is None:
        raw = load_from_directory()
    else:
        path = Path(source)
        if path.is_dir():
            raw = load_from_directory(path)
        elif path.suffix.lower() == ".zip":
            raw = load_from_zip(path)
        else:
            raise ValueError(f"Unsupported source: {source}")

    return clean(raw)
