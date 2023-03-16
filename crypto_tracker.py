import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

st.title("Crypto Price Tracker and Predictor")

st.sidebar.header("Select Crypto and Time Period")
crypto = st.sidebar.selectbox("Choose the cryptocurrency", ["LVL-USD","BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD"])
days_back = 90
days_ahead = 7

today = datetime.today()
start_date = today - timedelta(days=days_back)
end_date = today

data = yf.download(crypto, start=start_date, end=end_date)

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=data, x=data.index, y="Close", ax=ax)
ax.set(xlabel="Date", ylabel="Closing Price", title=f"{crypto} Price (Last {days_back} Days)")
st.pyplot(fig)

data["Day"] = np.arange(len(data))
X = data[["Day"]]
y = data[["Close"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"Prediction Accuracy (RMSE): {rmse:.2f}")
st.write(f"Prediction R-squared Score: {r2:.2f}")

future_days = np.arange(len(data), len(data) + days_ahead).reshape(-1, 1)
future_prices = model.predict(future_days)

future_dates = pd.date_range(end_date, periods=days_ahead + 1).tolist()[1:]
predictions = pd.DataFrame({"Date": future_dates, "Predicted Price": future_prices.flatten()})

st.write(f"Predicted {crypto} Prices for the Next {days_ahead} Days")
st.write(predictions)

# Plot the historical data and predictions
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.lineplot(data=data, x=data.index, y="Close", ax=ax2, label="Historical Prices")
sns.lineplot(x=future_dates, y=future_prices.flatten(), ax=ax2, label="Predicted Prices", color="red")
ax2.set(xlabel="Date", ylabel="Closing Price", title=f"{crypto} Price Prediction (Next {days_ahead} Days)")
ax2.legend()
st.pyplot(fig2)
