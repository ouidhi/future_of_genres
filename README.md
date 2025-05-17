# # Future of Music Genre Trends: Time Series Forecasting

## Overview
This project uses Google Trends data to forecast the popularity trends of 9 music genres from 2025 to 2026. Using SARIMA modeling, I aim to predict which genres will rise or fall in popularity, offering strategic insights for artists, record labels, and music marketers.

---

## Data Retrieval from Google Trends

I used the **PyTrends** library to fetch monthly search interest data from Google Trends for selected music genres from January 2018 through December 2024.

Since Google Trends API restricts queries to 5 search terms at a time, I retrieved the data in two batches and merged them:

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
```

## Dataset Description
- The resulting dataset contains monthly interest scores (0–100) for each genre.
- Columns represent genres, rows represent months from 2018 to 2024.
- Cleaned to remove low-variance columns (e.g., R&B was dropped due to insufficient variation).

## Project Workflow

1. Data Loading and Preprocessing
Imported the dataset and set the date column as a datetime index.
Selected relevant genre columns for analysis.

2. Exploratory Data Analysis (EDA)
Visualized trends to identify patterns.
Checked for missing data and handled anomalies.

3. Stationarity Check
Conducted Augmented Dickey-Fuller (ADF) tests on each genre's time series.
Stationarity is necessary for ARIMA/SARIMA models to perform well.

4. Seasonality Analysis
Used seasonal decomposition to identify seasonal trends (e.g., yearly cycles).
Seasonality justifies the choice of SARIMA over simpler ARIMA.

5. Train-Test Split
Split data into training (all data except last 12 months) and testing sets (last 12 months).
This allows for evaluation of model performance on unseen data.

6. Model Selection: Auto ARIMA
Utilized pmdarima.auto_arima to automatically select optimal p, d, q parameters.
Balanced between underfitting and overfitting.

7. SARIMA Modeling and Forecasting
Fitted SARIMA model using selected parameters.
Forecasted test set and evaluated model accuracy.
Produced final forecast for 24 months ahead (2025-2026).

8. Model Evaluation
Used Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to evaluate test forecasts.
Ensured model reliability before long-term forecasting.
Why SARIMA?

The dataset contains 83 monthly data points — enough to capture yearly seasonality.
SARIMA explicitly models both seasonal and non-seasonal components, critical for music trends that often show cyclic interest (e.g., seasonal releases, festival seasons).
Other models like ARIMA don’t account for seasonality as effectively.
Facebook Prophet is an alternative but SARIMA offers more statistical control and interpretability

