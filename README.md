# Future of Music Genre Trends: Time Series Forecasting

## 👁️‍🗨️ Why This Project Exists

The soundscape of global music is changing — faster than ever. TikTok virality, global genre crossovers, and streaming algorithms are reshaping what we listen to and why. But what if we could *predict* where the cultural current is headed next?

This project dives into that question by forecasting monthly search interest (2018–2024) for 9 major music genres and projecting their trajectories through 2025 and 2026. Think of it as a data crystal ball for music execs, brand strategists, and fans alike — spotlighting what genres are poised to surge, fade, or surprise us next.

---

## 📊 The Data: Where Culture Meets Curiosity

I used Google Trends data — a real-time signal of collective curiosity — because it’s not about *what people say they like,* but what they *actually look up*. It’s raw, reactive, and highly telling.

To collect the data:

* I used **PyTrends**, the unofficial Google Trends API, pulling monthly interest scores from January 2018 to December 2024.
* Due to API limitations (5 terms max), I divided the 9 genres into two batches and merged them for continuity.
* Final dataset: 9 genres × 84 months = 756 signals of shifting cultural taste.

### Dataset Description
- The resulting dataset contains monthly interest scores (0–100) for each genre.
- Columns represent genres, rows represent months from 2018 to 2024.
- Cleaned to remove low-variance columns (e.g., R&B was dropped due to insufficient variation).
  
```python
genres = ['hip hop music', 'pop music', 'r and b', 'country music', 'rock music',
          'alternative music', 'kpop music', 'metal music', 'latin music', 'indie music']
```

Genres like **R\&B** were removed due to low signal variance — a story in itself about fading public interest or the term not suitable enough to collect trend data. 

---

## 🧠 Behind the Modeling: Why SARIMA?

Music trends have rhythm — both metaphorically and statistically. That’s why I didn’t just pick any model — I used **SARIMA**, a time series model that embraces seasonality. We’re not just predicting *more of the same* — we’re mapping the *when* and *why* behind peaks and dips.

[Learn more about SARIMA.](Concepts.md#sarima-model-forecasting)

Each genre went through the same modeling workflow:

1. Stationarity checks via ADF tests.
2. Seasonality analysis to detect yearly rhythms.
3. Train/test splitting for real-world validation.
4. Auto-tuned SARIMA modeling (with manual diagnostics to avoid blind automation).
5. Forecast generation and visual storytelling.
6. Accuracy scoring and fine-tuning.


## 📊 Want to Skip the Process and Dive Into the Results?

If you're more interested in the story the data tells — not the modeling journey — you can jump straight into the final visualizations, forecasts, and genre insights.

👉 [Click here to explore the results.](Forecast_Results.md)

You'll find:

* 2-year forecasts for each genre
* Clear trends, turning points, and seasonality patterns
* Commentary on what the numbers reveal about music, culture, and what's next

## 🔎 Want to dig deeper into the full modeling process and genre-by-genre interpretations?

👉 [Check out the full notebook here.]()

This walks through each genre's stationarity, seasonality, model tuning, evaluation scores, and 24-month forecasts — everything from raw signal to cultural readout.

---

## 🔧 How I Built It

All code is written in Python, using:

* **pmdarima.auto\_arima** for parameter selection
* **SARIMAX** from `statsmodels` for modeling
* **Matplotlib** for storytelling through visuals
* **MAE, RMSE, MAPE** for evaluation
* Manual diagnostics like AIC, BIC, Ljung-Box and Jarque-Bera to ensure robust results

---

Here’s a teaser of what that looks like for Hip-Hop:

### **i. Is the data stationary?**

To build reliable forecasts using ARIMA or SARIMA, the data needs to be *stationary* — meaning its statistical properties (like mean and variance) don't shift over time.

I ran an Augmented Dickey-Fuller (ADF) test to check this. The result?

> **p-value = 0.296** (way above the 0.05 threshold)

So, no — Hip Hop's trend is **non-stationary**, which means we’ll need differencing in our model (hence the "I" in ARIMA).

---

### **ii. Does the data show seasonality?**

Yes, but it’s **weak**.

I performed a seasonal decomposition and saw a **repeating yearly pattern**, although not very strong. The seasonal strength score came out to **0.417** (closer to 0 = weak). Here's what the breakdown showed:

* **Trend:** A consistent downward trend from 2018 to 2022, then it flattens.
* **Seasonality:** Some repetition every year, but not super pronounced.
* **Residuals:** Random fluctuations — fairly balanced around zero, which is a good sign.

This justified the use of **SARIMA** (which accounts for both seasonality and non-stationarity) over simpler ARIMA.

[Learn more about Seasonality.](Concepts.md)

---

### **iii. How did I train and test the model?**

To simulate future forecasting conditions:

* I **trained** the model on all data **except the last 12 months**
* I **tested** the forecast performance on the **final year**

This split mimics how forecasts are used in real-world settings — we test predictions on unseen data.

---

### **iv. How did I pick the model parameters?**

Instead of guessing the (p, d, q) and seasonal values, I used `auto_arima()` to find the best-fitting parameters — but **with a twist**.

Before running it, I already:

* Checked for stationarity (to guide the value of `d`)
* Measured seasonal strength (to decide if seasonal terms were needed)

So this wasn't blind automation — it was **smart automation**, guided by the nature of the data.

> Final model: **SARIMA(1,1,1)(1,0,1)\[12]**

---

### **v. Forecasting: How well did it perform?**

Using the selected SARIMA model, I forecasted the next 12 months and compared predictions to the actual test data.

**Evaluation metrics:**

* **MAE:** 0.36 → average error size
* **RMSE:** 0.42 → no large error spikes
* **MAPE:** 7.5% → excellent accuracy (below 10% is strong)

The model also passed residual diagnostics:

* No autocorrelation (Ljung-Box p = 0.82)
* Residuals are normally distributed (JB p = 0.53)
* No heteroskedasticity (H p = 0.73)

![image](https://github.com/user-attachments/assets/aaba1016-6fd9-4d6c-98b4-229bff131cb6)

Visually, the forecast hugged the real values closely. ✅

---

### **vi. What does the future look like?**

With the model validated, I extended the forecast by **24 months** to cover 2025–2026.

![image](https://github.com/user-attachments/assets/ccdd11e7-9c16-4f40-abd3-605685b98254)

Key takeaways from the final forecast:

* The **downward trend** continues, but **slows down**
* Seasonal ups and downs persist — possibly tied to cultural/industry cycles
* Hip Hop popularity may be **stabilizing**, not crashing

---

### 🌟 Final Thoughts

Even Hip-Hop — a once-undisputed leader — shows a **downward trend** post-2021. Not a collapse, but a flattening curve... maybe making space for genres like Latin, K-Pop, or Indie to step up?

---

## 🎯 Who This Is For

* **Label execs & A\&R scouts** looking for the next big sound
* **Brand collab teams** identifying emerging music movements
* **Streaming platforms** shaping recommendation algorithms
* **Artists** curious about genre momentum & timing their releases

---

## 📦 Files in This Repo

* `trend_df.csv`: Cleaned Google Trends dataset (2018–2024)
* `forecasting.ipynb`: Full SARIMA modeling notebook (all genres)
* `Concepts.md`: In-depth breakdown of stationarity, seasonality, and SARIMA
* `Forecast_Results`: Folder with plots and forecast outputs

---

## ✨ Final Thought

This isn’t just data. It’s a mirror of culture — of what moves us, what fades, and what pulses just beneath the surface. Music isn’t static, and neither should our understanding of it be.

---

Want to collab or riff off this idea? Let’s chat: \[LinkedIn/Portfolio Link]
