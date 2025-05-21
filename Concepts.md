# < concepts guide >
> This section outlines the key concepts and theory behind the project, providing the necessary background to understand and interpret the time series analysis and forecasting techniques used.

## Time series analysis
Time series analysis = understanding the past.

Time series analysis is the process of examining time-ordered data points to identify patterns, trends, and other structures that can inform forecasting models. Understanding past behavior helps in constructing better predictive models.

###1. Plotting the series

The first step is always visual. By plotting the time series, we can spot overall patterns, trends, and potential anomalies. This step helps determine whether the data has a visible upward/downward trend, seasonal effects, or outliers.

2. Seasonality using decomposition

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


3. Checking if data is stationary

A stationary time series has constant statistical properties (mean, variance) over time. Most forecasting models (like ARIMA) assume the data is stationary.

I've checked for stationarity using:

Dickey-Fuller test (ADF test)- It is a statistical test used to check whether a time series is stationary — meaning its statistical properties like mean, variance, and autocorrelation don’t change over time.

*Hypotheses in ADF Test*

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


1. Selecting a model

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

2. Splitting training and test data

To validate performance, the dataset is split into:
- **Training set**: Used to build the model.
- **Test set**: Used to test its prediction power.

This mimics real-world forecasting where future data is unknown.

3. Evaluating the model performance

We assess the accuracy of predictions using metrics like:
-> **AIC, BIC**
  - These are model selection criteria that evaluate the goodness of fit while penalizing for model complexity.
  - Lower AIC and BIC values indicate a better-fitting model with fewer unnecessary parameters.
  
-> **MAE (Mean Absolute Error)**
  - MAE measures the average absolute difference between the predicted values and actual observations.
  - A lower MAE indicates that the model’s predictions are consistently close to real values.

-> **RMSE (Root Mean Squared Error)**
  - RMSE gives more weight to larger errors by squaring the deviations before averaging.
  - It is useful for identifying whether large prediction errors are present.

-> **MAPE (Mean Absolute Percentage Error)**
  - MAPE expresses the forecast error as a percentage of the actual values, which makes it scale-independent.
  - It’s particularly useful when comparing models across multiple genres with different popularity scales.

-> **Ljung-Box test**
  - The Ljung-Box test checks whether the residuals (forecast errors) are uncorrelated, meaning the model has captured all meaningful patterns.
  - A high p-value (p > 0.05) suggests no autocorrelation in residuals, indicating a well-fit model.

-> **Jarque-Bera test**
  - This test assesses whether the residuals follow a normal distribution, a key assumption for reliable confidence intervals.
  - A high p-value implies the residuals are approximately normal, validating the use of prediction intervals in forecasting.

-> **Forecast vs Actual Plot**
  - Visual comparison between the model’s forecast and the actual values is essential for intuitive evaluation.

Visual comparison between predicted and actual values is also crucial.

4. Creating a forecast for future 

After validation, the final model is trained on the full dataset to forecast future time periods.

## Time-series components

A given time series is thought to consist of three systematic components including level, trend, seasonality, and one non-systematic component called noise. 

## ARIMA/ SARIMA Model Forecasting

***ARIMA = AR + I + MA***

- AR (Auto-Regressive): Uses past values to predict future values.
- I (Integrated): Differencing to make data stationary.
- MA (Moving Average): Models the error of past forecasts.

***SARIMA = ARIMA + Seasonality***

SARIMA includes seasonal versions of AR, I, and MA components.

In this project, I’m analyzing how music genre popularity evolves over time using data from sources like Spotify and Google Trends. After exploring multiple time series forecasting methods, I selected SARIMA (Seasonal AutoRegressive Integrated Moving Average) as the primary model due to following reasons:

1. Captures Seasonality in Music Trends

Music consumption is heavily influenced by the time of year:
Pop often dominates in summer
Indie and alternative trends rise in fall

2. Supports Small-to-Medium Datasets

My genre-level popularity data spans 8-10 years on a monthly basis. That’s not huge — and deep learning models often underperform without large datasets.

3. Aligns with Decomposed Time Series

My analysis includes decomposition of the time series into trend, seasonality, and noise. SARIMA naturally incorporates these components within its structure, making it a seamless next step in the modeling pipeline.

### Components/ parameters of SARIMA 

These models are defined by the following parameters:

**ARIMA(p, d, q)**
- p: Number of lag observations (autoregressive terms).
- d: Degree of differencing (to remove trend and achieve stationarity).
- q: Size of moving average window.

SARIMA(p, d, q)(P, D, Q, s) adds:
- P: Seasonal autoregressive order
- D: Seasonal differencing order
- Q: Seasonal moving average order
- s: Length of the seasonal cycle (e.g., 12 for monthly data with yearly seasonality)

These components are generated using auto_arima() but I've used following measrues to fint-tune ht model for each genre.
- ACF, PACF plots
- Seasonality & seasonality strength
- Ljung-Box p-value
- Jarque-Bera p-value 

Sources

https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/ 
https://www.analyticsvidhya.com/blog/2018/08/auto-arima-time-series-modeling-python-r/ 
https://www.geeksforgeeks.org/sarima-seasonal-autoregressive-integrated-moving-average/ 

