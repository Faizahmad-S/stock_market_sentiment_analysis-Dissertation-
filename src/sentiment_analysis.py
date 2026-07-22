"""VADER and TextBlob sentiment classification.

Two lexicon-based sentiment models are exposed through a uniform interface:

    * :func:`vader_score` returns a compound score in ``[-1, 1]``.
    * :func:`textblob_score` returns a polarity score in ``[-1, 1]``.
    * :func:`classify` buckets a score into ``negative``/``neutral``/``positive``
      using the thresholds defined in :mod:`~src.utils`.

Convenience wrappers :func:`add_vader_columns` and :func:`add_textblob_columns`
apply the models to a whole DataFrame in place.

A single :func:`evaluate` helper computes accuracy, a classification report
and a confusion matrix against a labelled dataset (used to reproduce the
comparison against the Kaggle labels reported in the dissertation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .utils import NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD

# A single, lazily-initialised analyser is reused between calls. Instantiating
# the VADER lexicon is not free, so this matters when scoring thousands of
# headlines.
_vader_analyzer: SentimentIntensityAnalyzer | None = None


def _analyzer() -> SentimentIntensityAnalyzer:
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def vader_score(text: str) -> float:
    """Return the VADER compound score for ``text`` (``-1..1``)."""
    if not isinstance(text, str) or not text:
        return 0.0
    return _analyzer().polarity_scores(text)["compound"]


def textblob_score(text: str) -> float:
    """Return the TextBlob polarity score for ``text`` (``-1..1``)."""
    if not isinstance(text, str) or not text:
        return 0.0
    return TextBlob(text).sentiment.polarity


def classify(
    score: float,
    negative_threshold: float = NEGATIVE_THRESHOLD,
    positive_threshold: float = POSITIVE_THRESHOLD,
) -> str:
    """Bucket a numeric score into ``negative``/``neutral``/``positive``."""
    if score >= positive_threshold:
        return "positive"
    if score <= negative_threshold:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------
def add_vader_columns(
    df: pd.DataFrame, text_column: str = "Title", inplace: bool = False
) -> pd.DataFrame:
    """Add ``VADER_Sentiment`` (score) and ``VADER_Label`` (class) columns."""
    target = df if inplace else df.copy()
    target["VADER_Sentiment"] = target[text_column].apply(vader_score)
    target["VADER_Label"] = target["VADER_Sentiment"].apply(classify)
    return target


def add_textblob_columns(
    df: pd.DataFrame, text_column: str = "Title", inplace: bool = False
) -> pd.DataFrame:
    """Add ``TextBlob_Sentiment`` (score) and ``TextBlob_Label`` (class)."""
    target = df if inplace else df.copy()
    target["TextBlob_Sentiment"] = target[text_column].apply(textblob_score)
    target["TextBlob_Label"] = target["TextBlob_Sentiment"].apply(classify)
    return target


def add_sentiment_category(
    df: pd.DataFrame,
    score_column: str = "VADER_Sentiment",
    out_column: str = "Sentiment_Category",
    inplace: bool = False,
) -> pd.DataFrame:
    """Bucket a score column into ``Negative``/``Neutral``/``Positive``.

    Uses the fine-grained ``(-1, -0.05, 0.05, 1)`` bins from the dissertation
    (distinct from the coarser ``0.2`` thresholds used elsewhere).
    """
    target = df if inplace else df.copy()
    target[out_column] = pd.cut(
        target[score_column],
        bins=[-1, -0.05, 0.05, 1],
        labels=["Negative", "Neutral", "Positive"],
    )
    return target


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    """Container for the outputs of :func:`evaluate`."""

    accuracy: float
    report: str
    confusion_matrix: np.ndarray
    labels: list[str]

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy * 100:.2f}%\n"
            f"{self.report}"
        )


def _classifier(
    kind: str,
    negative_threshold: float,
    positive_threshold: float,
) -> Callable[[str], str]:
    scorer = vader_score if kind == "vader" else textblob_score

    def _fn(text: str) -> str:
        return classify(scorer(text), negative_threshold, positive_threshold)

    return _fn


def evaluate(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    kind: str = "vader",
    negative_threshold: float = NEGATIVE_THRESHOLD,
    positive_threshold: float = POSITIVE_THRESHOLD,
    labels: tuple[str, ...] = ("negative", "neutral", "positive"),
) -> EvaluationResult:
    """Score every row in ``df`` and compare against ``label_column``.

    Parameters
    ----------
    df:
        DataFrame containing headlines and true sentiment labels.
    text_column:
        Column containing the text to score.
    label_column:
        Column containing the ground-truth sentiment label.
    kind:
        ``"vader"`` or ``"textblob"``.
    """
    if kind not in {"vader", "textblob"}:
        raise ValueError("kind must be 'vader' or 'textblob'")

    classifier = _classifier(kind, negative_threshold, positive_threshold)
    predicted = df[text_column].apply(classifier)
    truth = df[label_column].astype(str).str.strip().str.lower()

    label_list = list(labels)
    return EvaluationResult(
        accuracy=accuracy_score(truth, predicted),
        report=classification_report(truth, predicted, labels=label_list, zero_division=0),
        confusion_matrix=confusion_matrix(truth, predicted, labels=label_list),
        labels=label_list,
    )
