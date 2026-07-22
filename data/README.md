# Data

This directory holds all input and intermediate data used by the project.

## Structure

```
data/
├── raw/                  # Daily scraped headline CSVs from Finviz
├── processed/            # Cleaned, merged, sentiment-scored output
├── Kaggle_dataset.csv    # Labelled sentiment corpus (evaluation only)
└── README.md
```

## `raw/`

The scraper (`src/scraper.py`) writes one CSV per run into this folder,
named `news_data_<timestamp>.csv`. Each file has the columns:

| Column  | Description                              |
|---------|------------------------------------------|
| Ticker  | Stock ticker symbol (e.g. `AAPL`)        |
| Date    | Publication date from Finviz             |
| Time    | Publication time from Finviz             |
| Title   | Headline text                            |

A `timestamp.txt` file is also written to record the most recent scrape.

## `processed/`

Output of `src.preprocessing.preprocess()` — the merged, cleaned and
chronologically sorted DataFrame, with sentiment columns added.
Regenerate with:

```python
from src.preprocessing import preprocess
from src.sentiment_analysis import add_vader_columns, add_textblob_columns

df = preprocess("data/raw")
df = add_vader_columns(df)
df = add_textblob_columns(df)
df.to_csv("data/processed/headlines_scored.csv", index=False)
```

## `Kaggle_dataset.csv`

Two-column labelled dataset (`sentiment,headline`) sourced from Kaggle,
used purely for accuracy evaluation of VADER and TextBlob. Encoding is
`ISO-8859-1`. Around 4,800 headlines with the class balance:

| Label     | Count |
|-----------|------:|
| neutral   | 2,879 |
| positive  | 1,363 |
| negative  |   604 |
