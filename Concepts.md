# < concepts guide >
> This section outlines the key concepts and theory behind the project, providing the necessary background to understand and interpret the time series analysis and forecasting techniques used.

## Time series analysis
Time series analysis = understanding the past.

Time series analysis is the process of examining time-ordered data points to identify patterns, trends, and other structures that can inform forecasting models. Understanding past behavior helps in constructing better predictive models.

**1. Plotting the series**

The first step is always visual. By plotting the time series, we can spot overall patterns, trends, and potential anomalies. This step helps determine whether the data has a visible upward/downward trend, seasonal effects, or outliers.

**2. Seasonality using decomposition**

Time series data is often composed of several distinct components:

- **Trend**: The long-term progression (upward or downward movement).
- **Seasonality**: Cyclical patterns that repeat at regular intervals (daily, monthly, yearly, etc.).
- **Residual/Noise**: The random variation or error after removing trend and seasonality.

Decomposition helps separate these components to better understand underlying structures in the data.

I've also used seasonal strength to back up the results. **Seasonal strength** tells us how much of the variability in our time series can be explained by its seasonal component (e.g., daily, weekly, or yearly patterns).

If seasonal strength is:
- Close to 1 → Strong, clear seasonal pattern
- Close to 0 → Weak or no seasonal pattern

```python
result = seasonal_decompose(genre_series, model='additive', period=12)  # for monthly data with yearly seasonality
result.plot()
plt.show()

# Seasonal strength
seasonal = result.seasonal
resid = result.resid

seasonal_strength = 1 - (np.var(resid.dropna()) / np.var((resid + seasonal).dropna()))
print("Seasonal Strength:", seasonal_strength)
```


**3. Checking if data is stationary**

A stationary time series has constant statistical properties (mean, variance) over time. Most forecasting models (like ARIMA) assume the data is stationary.

I've checked for stationarity using:

Dickey-Fuller test (ADF test)- It is a statistical test used to check whether a time series is stationary — meaning its statistical properties like mean, variance, and autocorrelation don’t change over time.

***Hypotheses in ADF Test***

- Null Hypothesis (H₀): The time series has a unit root → it's non-stationary.
- Alternative Hypothesis (H₁): The time series does not have a unit root → it's stationary.

*ADF Test Output & Interpretation*

```python
result = adfuller(genre_series)

print("ADF Statistic:", result[0])
print("p-value:", result[1])
```
We get two main things:
- ADF Statistic – A test statistic that’s compared to critical values.
- p-value – Tells you whether to reject the null hypothesis.

Interpretation of the p-value
- p-value < 0.05 → Reject the null hypothesis → Stationary series.
- p-value > 0.05 → Fail to reject the null → Non-stationary series.


If the series is not stationary, techniques like differencing, log transformation, or seasonal adjustment are commonly applied to induce stationarity. 

While I'm using auto_arima() to automatically determine the optimal values for p, d, q, the results from the ADF test can be a valuable reference — especially for selecting the appropriate differencing term d. Since auto_arima() uses statistical heuristics and may not always capture the true dynamics of the data, it's good practice to validate or tweak its suggestions based on domain knowledge and stationarity test results like ADF. Same can be said for tweaking the seasonal components using seasonality. 


## Time series forecasting

Time series forecasting = predicting the future.

After analyzing past data, the next goal is to use that information to forecast future values. This involves selecting a model, evaluating its accuracy, and generating future predictions. This includes following steps:


**1. Selecting a model**

Different Kinds of Time Series Forecasting Models:

- **Naive Forecasting**: Assumes future values equal the last observed value.
- **Moving Average / Exponential Smoothing**: Averages out noise, gives weight to recent data.
- **ARIMA (AutoRegressive Integrated Moving Average)**: Good for non-seasonal data.
- **SARIMA (Seasonal ARIMA)**: Extension of ARIMA for seasonal datasets.
- **Prophet (by Facebook)**: Handles missing data, seasonality, and holidays.
- **LSTM (Neural Networks)**: For deep learning approaches to time series.

Each model has pros and cons based on the complexity and behavior of your data.

The choice depends on:
- Presence of seasonality
- Stationarity
- Complexity of the data

**2. Splitting training and test data**

To validate performance, the dataset is split into:
- **Training set**: Used to build the model.
- **Test set**: Used to test its prediction power.

This mimics real-world forecasting where future data is unknown.

**3. Evaluating the model performance**

We assess the accuracy of predictions using metrics like:

> **AIC, BIC**
  - These are model selection criteria that evaluate the goodness of fit while penalizing for model complexity.
  - Lower AIC and BIC values indicate a better-fitting model with fewer unnecessary parameters.
  
> **MAE (Mean Absolute Error)**
  - MAE measures the average absolute difference between the predicted values and actual observations.
  - A lower MAE indicates that the model’s predictions are consistently close to real values.

> **RMSE (Root Mean Squared Error)**
  - RMSE gives more weight to larger errors by squaring the deviations before averaging.
  - It is useful for identifying whether large prediction errors are present.

> **MAPE (Mean Absolute Percentage Error)**
  - MAPE expresses the forecast error as a percentage of the actual values, which makes it scale-independent.
  - It’s particularly useful when comparing models across multiple genres with different popularity scales.

> **Ljung-Box test**
  - The Ljung-Box test checks whether the residuals (forecast errors) are uncorrelated, meaning the model has captured all meaningful patterns.
  - A high p-value (p > 0.05) suggests no autocorrelation in residuals, indicating a well-fit model.

> **Jarque-Bera test**
  - This test assesses whether the residuals follow a normal distribution, a key assumption for reliable confidence intervals.
  - A high p-value implies the residuals are approximately normal, validating the use of prediction intervals in forecasting.

> **Forecast vs Actual Plot**
  - Visual comparison between the model’s forecast and the actual values is essential for intuitive evaluation.

**4. Creating a forecast for future**

After validation, the final model is trained on the full dataset to forecast future time periods. These factors are also used to tweak the parameters of the model. 

## ARIMA Model Forecasting

*ARIMA = AR + I + MA*

ARIMA stands for AutoRegressive Integrated Moving Average — a classic and powerful time series forecasting method.

It's built on three main components:
- AR (AutoRegressive): Uses the relationship between an observation and a number of lagged observations (i.e., past values).
- I (Integrated): Applies differencing to remove trends and make the time series stationary (where the mean and variance stay constant over time).
- MA (Moving Average): Models the error of the past forecast as part of the prediction equation.

ARIMA(p, d, q) includes:
- p (AutoRegressive): How many past values should influence the current value?
- d (Integrated): How many times should we difference the data to make it stationary (i.e., stable over time)?
- q (Moving Average): How many past forecast errors should we include?

## SARIMA Model Forecasting

*SARIMA = ARIMA + Seasonality*

SARIMA (Seasonal ARIMA) is an extension of ARIMA that accounts for seasonality — repeating patterns that occur at regular intervals (monthly, quarterly, etc.).

Most real-world time series are not only influenced by their recent past (ARIMA), but also by seasonal patterns — like how certain genres spike during holidays or how moods change by season (e.g., upbeat pop in summer, chill indie in fall).

SARIMA captures both:
- Short-term trends: "What just happened?"
- Seasonal cycles: "What usually happens this time of year?"

SARIMA(p, d, q)(P, D, Q, s) adds:
🔵 Non-Seasonal Part: ARIMA(p, d, q)
- Same as ARIMA. 

🟣 Seasonal Part: (P, D, Q, s)
- P (Seasonal AutoRegressive): How many seasonal lags (e.g., 12 months ago) should influence the current value?
- D (Seasonal Differencing): How many times should we remove seasonal trends (like year-over-year patterns)?
    - Differencing removes repeated seasonal structures. Let’s say you sell more cold drinks every summer. That’s not a       random trend — it’s seasonal. Seasonal differencing helps isolate this pattern so the model can learn it and            adjust its forecasts accordingly. Without it, you'd confuse a seasonal pattern with a long-term trend — which           would throw off your model.
- Q (Seasonal Moving Average): How many past seasonal errors should be included?
- s (Seasonal Period): The length of the seasonal cycle (e.g., 12 for yearly cycles in monthly data).


**Why SARIMA?**

In this project, I’m analyzing how music genre popularity evolves over time using data from sources like Spotify and Google Trends. After exploring multiple time series forecasting methods, I selected SARIMA (Seasonal AutoRegressive Integrated Moving Average) as the primary model due to following reasons:

- Music genre popularity isn't random — it often has a seasonal rhythm:
  - Pop spikes in the summer
  - Acoustic/indie gains traction in autumn
- EDM and hip-hop surge in festival and party seasons
- This seasonality is clearly visible in the decomposition plots, and quantified using seasonal strength. SARIMA is perfect for this kind of data because:
  - It captures both long-term trends and seasonal fluctuations
  - It's suitable for moderate-sized datasets
  - It works well after decomposition and stationarity checks

**How I used SARIMA?**

To train and forecast with SARIMA:
- I used historical data on genre popularity (monthly)
- Checked for seasonality and stationarity
- Split the data into training and test datasets.
- Used auto_arima() for parameter suggestions
- Fine-tuned the model based on seasonal strength, ADF test, and residual diagnostics.
- Then I trained the SARIMA model like this:

```python
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
```
- Evaluated the model
- Created the unseen forecast for the next 2 years. 

Sources

https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/ 
https://www.analyticsvidhya.com/blog/2018/08/auto-arima-time-series-modeling-python-r/ 
https://www.geeksforgeeks.org/sarima-seasonal-autoregressive-integrated-moving-average/ 

