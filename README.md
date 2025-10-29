# Stock Market Prediction (LSTM)

## 1. Project Overview
This project trains a Recurrent Neural Network (Long Short-Term Memory / LSTM) model to predict future stock prices using historical market data and technical indicators.

Pipeline:
- Load historical daily stock data (`WYNN.csv`)
- Engineer indicators (moving averages, MACD, volatility, etc.)
- Create time-window sequences
- Train an LSTM to predict the next day's price
- Visualize actual vs predicted prices

---

## 2. Data and Features

### Input columns from CSV
- `Date`
- `Open`, `High`, `Low`, `Close`, `Adj Close`
- `Volume`

### Engineered features
- `Month`, `Year`
- `SMA_20`: 20-day Simple Moving Average of `Adj Close`
- `EMA_20`: 20-day Exponential Moving Average of `Adj Close`
- `MACD`: `EMA_26 - EMA_12`
- `MACD_Signal`: 9-day EMA of MACD
- `Volatility_10`: 10-day rolling standard deviation of `Adj Close`

After cleaning and dropping unused columns, the model trains on:
- `Close`
- `Adj Close`
- `Volume`
- `EMA_20`
- `MACD`
- `Volatility_10`

We drop:
- `Open`, `High`, `Low`, `SMA_20`, `MACD_Signal`
- Time columns (`Date`, `Month`, `Year`) before feeding to the model
- Rows with `NaN` from rolling windows

---

## 3. Train / Validation / Test Split
Data is split chronologically (no shuffling):
- Training set: first 70 percent
- Validation set: next 15 percent
- Test set: final 15 percent

```text
train_df = df_reg.iloc[:n_train]
val_df   = df_reg[n_train:n_val]
test_df  = df_reg[n_val:]
```

This preserves time order, which is critical for forecasting.

---

## 4. Model Architecture
We use a stacked LSTM regression model:

```python
model1 = Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(window, X_train.shape[2])),
    layers.Dropout(0.2),
    layers.LSTM(64),
    layers.Dense(1)
])
model1.compile(loss='mse', optimizer='adam')
```

Details:
- Input shape: `(window=60 timesteps, num_features=6)`
- Output: next-day price (regression)
- Loss: Mean Squared Error
- Optimizer: Adam
- Trained briefly for demonstration:
  ```python
  history1 = model1.fit(
      X_train, y_train,
      validation_data=(X_val, y_val),
      epochs=1,
      batch_size=1
  )
  ```

---

## 5. Visualization
We generate:
- Correlation heatmaps (feature relationships)
- Feature time series plots (Plotly line charts)
- Actual vs Predicted comparison:
  - Training actual prices
  - Validation actual prices
  - Validation predicted prices from the LSTM

We also plot validation period forecasts using a candlestick-style view plus the model’s predicted curve to visually inspect alignment.

---

## 6. Key Notes
- We predict the next day's price using the previous 60 days of multivariate signals.
- Features include price levels, volume, trend indicators (EMA, MACD), and short-term volatility.
- Data split is strictly forward in time to avoid leakage.
- The model currently runs for 1 epoch and batch size 1; this is a baseline, not a final tuned model.
