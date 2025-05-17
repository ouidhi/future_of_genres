# # Future of Music Genre Trends: Time Series Forecasting

## Overview
This project uses Google Trends data to forecast the popularity trends of 9 music genres from 2025 to 2026. Using SARIMA modeling, I aim to predict which genres will rise or fall in popularity, offering strategic insights for artists, record labels, and music marketers.

---

## Data Retrieval from Google Trends

We used the **PyTrends** library to fetch monthly search interest data from Google Trends for selected music genres from January 2018 through December 2024.

Since Google Trends API restricts queries to 5 search terms at a time, we retrieved the data in two batches and merged them:

```python
import pandas as pd
from pytrends.request import TrendReq
import time

# Genre search terms
genres = ['hip hop music', 'pop music', 'r and b', 'country music', 'rock music',
          'alternative music', 'kpop music', 'metal music', 'latin music', 'indie music']

# Connect to Google Trends
pytrends = TrendReq()

# Query first 5 genres (2018 to 2024)
pytrends.build_payload(genres[:5], timeframe='2018-01-01 2024-12-31')
time.sleep(300)  # delay to respect API limits
trend_df1 = pytrends.interest_over_time()
trend_df1.drop(columns=['isPartial'], inplace=True)

# Query last 5 genres
pytrends.build_payload(genres[5:], timeframe='2018-01-01 2024-12-31')
time.sleep(200)
trend_df2 = pytrends.interest_over_time()
trend_df2.drop(columns=['isPartial'], inplace=True)

# Merge the two dataframes
trend_df = trend_df1.join(trend_df2)
trend_df.dropna(inplace=True)

# Save to CSV for further analysis
trend_df.to_csv('trend_df.csv', index=True) 
Tracks how music genres evolve and predicts future trends using Spotify + Google Trends data. Also explores if genre diversity boosts artist popularity with entropy scores and regression. Insightful for labels, A\&amp;R, and marketing teams spotting emerging genres and versatile artists.
