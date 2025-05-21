# Future of Music Genre Trends: Time Series Forecasting

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
```
Tracks how music genres evolve and predicts future trends using Spotify + Google Trends data. Also explores if genre diversity boosts artist popularity with entropy scores and regression. Insightful for labels, A&R, and marketing teams spotting emerging genres and versatile artists.

## Dataset Description
- The resulting dataset contains monthly interest scores (0–100) for each genre.
- Columns represent genres, rows represent months from 2018 to 2024.
- Cleaned to remove low-variance columns (e.g., R&B was dropped due to insufficient variation).

## Project Workflow

1. **Importing libraries, Data Loading and Preprocessing**

- Imported the dataset and set the date column as a datetime index.
- Selected relevant genre columns for analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima # chooses best value for p,d,q 
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# time series data
df = pd.read_csv('/Users/vidhiparmar/Desktop/projects/future_genre/trend_data/trend_df.csv')

# preprocessing the dataframe
df.drop('r and b music', axis= 1, inplace= True) # not sufficient variation in the column - r and b music
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace= True)
```

2. **Exploratory Data Analysis (EDA)**

- First 5 records. 
- Checked for missing data and handled anomalies.
- Extracted each genre as a Pandas Series object. 

```python
print(f'Top 5 rows:\n{df.head()}\n')
print(f'Summary Stats:\n{df.describe()}\n')
print(f'Missing null values:\n{df.isna().sum()}')

hiphop = df['hip hop music']
pop = df['pop music']
country = df['country music']
rock = df['rock music']
alter = df['alternative music']
kpop = df['kpop music']
metal = df['metal music']
latin = df['latin music']
indie = df['indie music']
```

3. **For each genre, following steps were taken to create a 2 year forecast.**

*I have shown the detailed process for the genre Hip-hop here but the same procedure is used for the other 8 genres.*

**i. Stationarity Check**

- Conducted Augmented Dickey-Fuller (ADF) tests on each genre's time series.
- Stationarity is necessary for ARIMA/SARIMA models to perform well.

Learn more about stationaity in time series. 

```python
# Stationarity check 
result = adfuller(hiphop)

print("ADF Statistic:", result[0])
print("p-value:", result[1])
```
Output:
- ADF Statistic: -1.9786153985808255
- p-value: 0.2960274892539917

the p-value is 0.296, which is much higher than 0.05, so we conclude that the 'hiphop' time series is non-stationary. This means that the mean, variance, or both are not constant over time, which can affect modeling techniques like ARIMA that assume stationarity.

**ii. Seasonality Analysis**

- Used seasonal decomposition to identify seasonal trends (e.g., yearly cycles).
- Seasonality justifies the choice of SARIMA over simpler ARIMA.


```python
result = seasonal_decompose(hiphop, model='additive', period=12)  # for monthly data with yearly seasonality
result.plot()
plt.show()

# Seasonal strength
seasonal = result.seasonal
resid = result.resid

seasonal_strength = 1 - (np.var(resid.dropna()) / np.var((resid + seasonal).dropna()))
print("Seasonal Strength:", seasonal_strength)
```
![hiphop_season](https://github.com/user-attachments/assets/8f9ae497-7519-4656-bed5-ba3e70b70dab)

This plot shows the decomposition of the time series into Trend, Seasonality, and Residuals (Noise):

**Top plot:**
This is the original time series. There is a gradual decline in values from 2018 to 2024, suggesting a downward trend in hip hop popularity. 

**Trend:**
A clear, steady decline is visible from 2018 to around 2022, and then it flattens out. This tells us that the overall interest or metric is dropping over time.

**Seasonal:**
This part repeats with a regular yearly pattern, suggesting strong seasonal cycles—probably peaks and drops at specific months every year.

**Residuals:**
These are the random fluctuations left after removing the trend and seasonality. They're spread relatively evenly around zero, which is good—it means the model captured the trend and seasonality reasonably well.

Seasonal strength is a quantitative measure of how strong the seasonal component is in out data.

- If seasonality is strong, seasonal_strength close to 1.
- If seasonality is weak or absent, seasonal_strength close to 0.

Here, it is 0.417 which is close to 0. Hence, there is weak seasonality but it is present. 

Learn more about Seasonality. 

**iii. Train-Test Split**

Split data into training (all data except last 12 months) and testing sets (last 12 months).

This allows for evaluation of model performance on unseen data.

```python
train = hiphop.iloc[:-12]
test = hiphop.iloc[-12:]
```

**iv. Model Selection using auto_arima**  

Utilized pmdarima.auto_arima to automatically select optimal p, d, q parameters.

To improve the accuracy and reliability of the auto_arima() model selection, I first conducted a stationarity check using the Augmented Dickey-Fuller (ADF) test, and calculated the seasonal strength of the time series. These steps provided critical insights into the nature of the data — particularly the presence of a unit root (non-stationarity) and the influence of seasonal patterns.

While auto_arima() is a powerful tool for automating the selection of (p, d, q) and seasonal components, it can sometimes overfit or underfit the model — especially when the data has borderline stationarity or ambiguous seasonality. By using the ADF results to validate or adjust the differencing term d, and leveraging the seasonal strength to assess the need for seasonal parameters, I was able to fine-tune the model more effectively.

This hybrid approach — combining automated selection with domain-aware diagnostics — helps ensure the final ARIMA model is both statistically sound and well-suited to the structure of the data.


Learn more about ARIMA/ SARIMA modelling.

```python
# chooses the best values for p,d,q 
auto_model = auto_arima(train, seasonal=True, m=12, trace=True)
print(auto_model.summary())
```
Output:

Best model:  ARIMA(1,1,1)(1,0,1)[12] intercept

- **ARIMA(1,1,1)** → The non-seasonal part of the model has:
- **Seasonal Order (1, 0, 1, 12)** → The seasonal part of the model has:

I will now fit the SARIMA model using these parameters.

**v. SARIMA Modeling and Forecasting**

- Fitted SARIMA model using selected parameters.
- Forecasted test set and evaluated model accuracy.

```python
# fitting a SARIMA model by adjusting the parameters by evaluation the model   
model = SARIMAX(train, 
                order=(1, 1, 1), # (p, d, q)
                seasonal_order=(1, 0, 1, 12), # (P, D, Q, m)
                enforce_stationarity=False, 
                enforce_invertibility=False)

model_fit = model.fit()
print(model_fit.summary())

# forecasting the next 12 months using the trained model.
forecast = model_fit.get_forecast(steps=12)
hiphop_forecast = forecast.predicted_mean
conf_int = forecast.conf_int()

# creating a future date index for plotting the forecast.
future_dates = pd.date_range(start=train.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

# plotting the historical data, forecasted values, and the confidence intervals
plt.figure(figsize=(12, 5))
plt.plot(train, label='Historical')
plt.plot(future_dates, hiphop_forecast, label='Forecast', color='pink')
plt.fill_between(future_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.2)
plt.title('SARIMA Forecast')
plt.legend()
plt.tight_layout()
plt.show()
```
Output:

<img width="678" alt="Screenshot 2025-05-17 at 4 01 39 PM" src="https://github.com/user-attachments/assets/40670769-01cc-4e90-bc7f-62170aea8133" />

![image](https://github.com/user-attachments/assets/15b263f1-eb7e-47e5-8d21-536ebe7dc0d4)


**vi. Model Evaluation**

Used Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE) to evaluate test forecasts. 

Also using the Ljung-Box p-value, Jarque-Bera p-value, the stationarity and seasonality checks to fine-tune the model.  

```python
# Actual and predicted values
y_true = test
y_pred = forecast.predicted_mean

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast.predicted_mean, label='Forecast')
plt.legend()
```

Ensured model reliability before long-term forecasting.

**vii. Final 2 year forecast**

Produced final forecast for 24 months ahead (2025-2026).

```python
# fitting a SARIMA model by adjusting the parameters by evaluation the model   
model = SARIMAX(hiphop, 
                order=(1, 1, 1), # (p, d, q)
                seasonal_order=(1, 0, 1, 12), # (P, D, Q, m)
                enforce_stationarity=False, 
                enforce_invertibility=False)

model_fit = model.fit()
print(model_fit.summary())

# forecasting the next 24 months using the trained and tested model.
forecast = model_fit.get_forecast(steps=24)
mean_forecast = forecast.predicted_mean
conf_int = forecast.conf_int()

# creating a future date index for plotting the forecast.
future_dates = pd.date_range(start=hiphop.index[-1] + pd.DateOffset(months=1), periods=24, freq='MS')


# plotting the historical data, forecasted values, and the confidence intervals
plt.figure(figsize=(12, 5))
plt.plot(hiphop, label='Historical')
plt.plot(future_dates, mean_forecast, label='Forecast', color='pink')
plt.fill_between(future_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.2)
plt.title('SARIMA Forecast')
plt.legend()
plt.tight_layout()
plt.show()
```

## Why SARIMA?

The dataset contains 83 monthly data points — enough to capture yearly seasonality.
SARIMA explicitly models both seasonal and non-seasonal components, critical for music trends that often show cyclic interest (e.g., seasonal releases, festival seasons).

Other models like ARIMA don’t account for seasonality as effectively.

Facebook Prophet is an alternative but SARIMA offers more statistical control and interpretability

