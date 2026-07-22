"""Stock Market Sentiment Analysis package.

Modules
-------
scraper
    Scrape financial news headlines from Finviz.
preprocessing
    Merge, clean and prepare the scraped headline CSVs.
sentiment_analysis
    VADER and TextBlob sentiment classification.
stock_data
    Historical price retrieval via yfinance.
visualization
    Reusable plotting functions used throughout the analysis.
utils
    Shared constants and helpers.
"""

__version__ = "1.0.0"
__author__ = "Faizanahmad Sangolli"
