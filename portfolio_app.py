# Import necessary libraries
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Set the title of the app
st.title("Portfolio Visualization test")

# Ticker selection in sidebar
tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'META', 'GOOGL']
selected_tickers = st.sidebar.multiselect("Select tickers", tickers, default=tickers)

# Date selection in sidebar
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# Download data from yfinance
data = yf.download(selected_tickers, start=start_date, end=end_date)['Adj Close']
st.write("Stock Price Data", data.head())

# Calculate daily log returns
log_returns = np.log(data / data.shift(1))
st.write("Log Returns", log_returns.head())

# Define the portfolio performance function
def portfolio_performance(weights, log_returns):
    expected_return = np.sum(weights * log_returns.mean()) * 252  # 252 trading days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
    return expected_return, portfolio_std

# Number of portfolios to simulate
num_portfolios = 10000
results = np.zeros((3, num_portfolios))

# Simulate portfolios
for i in range(num_portfolios):
    weights = np.random.random(len(selected_tickers))
    weights /= np.sum(weights)
    expected_return, portfolio_std = portfolio_performance(weights, log_returns)
    results[0, i] = portfolio_std
    results[1, i] = expected_return
    results[2, i] = expected_return / portfolio_std  # Sharpe ratio

# Plot the simulated portfolios
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis')
plt.colorbar(scatter, label='Sharpe Ratio')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return')
plt.title('Simulated Portfolios')

# Display the plot in Streamlit
st.pyplot(fig)
