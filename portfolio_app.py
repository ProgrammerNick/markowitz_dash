# Import necessary libraries
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the title of the app
st.title("Portfolio Visualization with Dividends")

# Ticker selection in sidebar
tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'META', 'GOOGL', 'QCOM', 'ISTB']
selected_tickers = st.sidebar.multiselect("Select tickers", tickers, default=tickers)

# Date selection in sidebar
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

date_diff = end_date - start_date
years_diff = date_diff.days / 365.25

# Download adjusted close price data from yfinance
data = yf.download(selected_tickers, start=start_date, end=end_date)['Adj Close']
st.write("Stock Price Data", data.head())

# Download dividends data from yfinance and calculate dividend yield
dividends_data = pd.DataFrame()
for ticker in selected_tickers:
    ticker_dividends = yf.download(ticker, start=start_date, end=end_date, actions=True)['Dividends']
    dividends_data[ticker] = ticker_dividends
dividends_data = dividends_data.fillna(0)  # Replace NaN with 0 for days with no dividends

# Calculate daily log returns
log_returns = np.log(data / data.shift(1))
st.write("Log Returns", log_returns.head())

# Calculate annual dividend yield
dividend_yields = dividends_data.sum() / data.mean()
annual_dividend_yield = dividend_yields / years_diff

# Define the portfolio performance function with dividends
def portfolio_performance(weights, log_returns, dividend_yields):
    # Expected return includes both capital gains and dividend yield
    expected_return = np.sum(weights * (log_returns.mean() * 252 + dividend_yields))
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
    return expected_return, portfolio_std

# Number of portfolios to simulate
num_portfolios = 10000
results = np.zeros((3, num_portfolios))

tracked_weights = []
tracked_returns = []

# Simulate portfolios
for i in range(num_portfolios):
    weights = np.random.random(len(selected_tickers))
    weights /= np.sum(weights)
    tracked_weights.append(weights)
    expected_return, portfolio_std = portfolio_performance(weights, log_returns, annual_dividend_yield)
    tracked_returns.append(expected_return)
    results[0, i] = portfolio_std
    results[1, i] = expected_return
    results[2, i] = expected_return / portfolio_std  # Sharpe ratio

# Plot the simulated portfolios
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis')
plt.colorbar(scatter, label='Sharpe Ratio')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return')
plt.title('Simulated Portfolios with Dividends')

# Display the plot in Streamlit
st.pyplot(fig)
best_weights = sorted(zip(results[2], tracked_weights, tracked_returns), key=lambda x: x[0], reverse=True)[:5]

for sharpe, weights, returns in best_weights:
            st.write(f"Sharpe Ratio: {sharpe:.4f}")
            st.write(f"Expected Return: {returns:.4f}")
            weights_df = pd.DataFrame({"Ticker": sorted(selected_tickers), "Weight": weights})
            st.dataframe(weights_df)
            st.write("---")
