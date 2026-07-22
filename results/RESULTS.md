# Results

This document summarises the findings of the stock-market sentiment
analysis. It draws on financial news headlines scraped from Finviz for four
tickers (AAPL, AMZN, GOOG, WMT) and a labelled Kaggle dataset of ~4,800
financial headlines used to benchmark the two sentiment models.

All figures referenced below are in the [`images/`](../images) folder.
Raw metric outputs are in this folder:
[`vader_classification_report.txt`](vader_classification_report.txt),
[`textblob_classification_report.txt`](textblob_classification_report.txt),
[`aapl_daily_sentiment.csv`](aapl_daily_sentiment.csv).

---

## 1. Sentiment of scraped headlines

**Distribution of scores** (`fig01_vader_score_distribution.png`).
VADER compound scores cluster heavily around 0. Most financial headlines are
factual and neutral in tone; positive scores are moderate (concentrated
around 0.25–0.5) and strong negative scores (below −0.5) are rare. This is
consistent with financial reporting favouring objectivity over emotive
language.

**Comparison across stocks** (`fig02_sentiment_comparison_stocks.png`).
All four tickers have a median sentiment slightly above zero. Apple and
Amazon show the widest spread and the most negative outliers, reflecting
their heavier and more varied news coverage. Google is more stable, and
Walmart is the most compact around neutral.

**Sentiment categories** (`fig03_sentiment_categories.png`).
When bucketed at ±0.05, the coverage skews positive/neutral, with negatives
the smallest group — again pointing to a generally measured tone in the
sampled financial news.

**Coverage volume** (`fig04_articles_per_ticker.png`).
Apple received the most articles (250+), followed closely by Amazon and
Google, with Walmart the least (~150). The imbalance matters: tickers with
fewer articles produce less stable daily sentiment averages, so results for
Walmart should be read with more caution than those for Apple.

**Word cloud** (`fig05_wordcloud.jpeg`).
"Apple", "Amazon" and "Stock" dominate the headline vocabulary, alongside
market terms ("Market", "Buy", "Deal", "Growth") and tech themes ("AI",
"iPhone", "Nvidia"). Neutral reporting verbs ("say", "according", "report")
are also prominent, reinforcing the neutral-tone finding.

## 2. Sentiment over time

**Daily average** (`fig06_average_sentiment_per_day.png`).
Aggregate sentiment fluctuates day to day. Peaks appear around 6 and 10
September; the clearest negative dip is on 4 September.

**Apple trend** (`fig07_aapl_sentiment_trend.png`).
Apple sentiment starts positive (~0.2 on 11 Sep), stays positive through
mid-month, then declines from ~15 September onward, turning negative by
21 September (−0.22). This is the sharpest single-ticker swing in the window.

**Heat map** (`fig08_sentiment_heatmap.png`).
Across the period, most cells are neutral-to-green (positive). The one
strong red cell is Apple on 21 September. Amazon and Google are steady;
Walmart is consistently light-green (positive).

**Per-ticker by date** (`fig09_mean_sentiment_per_ticker.png`).
Confirms the same pattern: AAPL and AMZN show the largest positive and
negative extremes, while GOOG and WMT stay within a narrower band.

## 3. Model accuracy (VADER vs TextBlob)

Both models were run against the labelled Kaggle dataset
(604 negative, 2,879 neutral, 1,363 positive; 4,846 total).

| Model    | Overall accuracy | Neutral F1 | Positive F1 | Negative F1 |
|----------|-----------------:|-----------:|------------:|------------:|
| VADER    | **55.03 %**      | 0.62       | 0.51        | **0.32**    |
| TextBlob | **58.53 %**      | **0.72**   | 0.32        | 0.13        |

*(Reproduced values; see `fig10_vader_accuracy_report.png` and
`fig12_textblob_accuracy_report.png` for the full per-class reports, and
`fig11_vader_confusion_matrix.png` / `fig13_textblob_confusion_matrix.png`
for the confusion matrices. `fig_model_accuracy_comparison.png` shows the
headline accuracy comparison.)*

**Key takeaways:**

- **TextBlob has the higher headline accuracy** (~3.5 points), but this is
  driven almost entirely by the majority neutral class — it over-classifies
  headlines as neutral and collapses on negatives (recall 0.08, F1 0.13).
- **VADER is more balanced.** It handles the polar classes far better
  (negative F1 0.32 vs 0.13; positive F1 0.51 vs 0.32), which matters more
  for a task where the *actionable* signal is positive vs negative news.
- Both models are strongest on neutral sentiment and weakest on negative —
  a known limitation of general-purpose lexicons applied to financial text.

**For this reason VADER was chosen as the primary model** for the Apple case
study, despite its slightly lower headline accuracy.

## 4. Sentiment vs stock price (Apple case study)

`fig14_15_aapl_sentiment_vs_price.png` places Apple's daily mean VADER
sentiment next to its opening/closing prices for 10–22 September 2024
(daily sentiment values in `aapl_daily_sentiment.csv`).

- **Positive phase (10–15 Sep):** sentiment stays positive (peak 0.257 on
  12 Sep) and prices trend upward — sentiment and price move together.
- **Decline (16–19 Sep):** sentiment weakens (turning negative on 16 Sep)
  and prices fall to their low for the window — the correlation holds.
- **Recovery (18–22 Sep):** prices rebound sharply *despite* sentiment
  staying negative (low of −0.22 on 21 Sep). This coincides with the
  **iPhone 16 launch on 20 September 2024**, a real-world catalyst that
  moved the stock independently of headline sentiment.

**Interpretation.** There is a clear *short-term* correlation between news
sentiment and price direction, but sentiment is not the sole driver. A major
product event overrode negative headline sentiment at the end of the window.
Sentiment is best treated as one input among several, not a standalone
predictor.

## 5. Limitations

- Both models are lexicon-based and general-purpose; they miss financial
  jargon, sarcasm and complex phrasing, and default to neutral under
  ambiguity.
- Only headlines are scored, not article bodies.
- The analysis window is short (~2 weeks), so only short-term effects are
  captured.
- Coverage is imbalanced across tickers, and comes from a single source
  (Finviz).

## 6. How these outputs were produced

- **Figures 10–13** and both classification reports are regenerated exactly
  from `data/Kaggle_dataset.csv` via `src/sentiment_analysis.evaluate()`.
  Running the pipeline locally reproduces VADER at 55.04 % and the identical
  per-class numbers reported above.
- **Figures 1–9** depend on the daily scraped headline CSVs. Once you have
  accumulated raw files in `data/raw/`, they regenerate from
  `src/visualization.py` (the plotting functions are named to match each
  figure). The versions shown here are the authoritative ones from the
  dissertation, since the original scraped batch was not retained.
- **Figure 14/15** uses the retained Apple daily-sentiment series plus live
  prices from `yfinance` (`src/stock_data.fetch_prices`).
