# Import necessary libraries 
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Set the title of the app
st.title("Portfolio Visualization with Dividends")

# Ticker selection in sidebar
tickers = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'META', 'GOOGL']
selected_tickers = st.sidebar.multiselect("Select tickers", tickers, default=tickers)

# Date selection in sidebar
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# Download adjusted close prices and dividends
data = yf.download(selected_tickers, start=start_date, end=end_date)
price_data = data['Adj Close']
dividend_data = data['Dividends']

# Calculate daily log returns for price data
log_returns = np.log(price_data / price_data.shift(1))

# Calculate dividend yield
dividend_yield = dividend_data / price_data.shift(1)

# Nicolas new function: Calculate total returns including dividends
def calculate_total_returns(log_returns, dividend_yield):
    total_returns = log_returns + dividend_yield
    return total_returns

# Use the new function to calculate total returns
total_returns = calculate_total_returns(log_returns, dividend_yield)
st.write("Total Returns with Dividends", total_returns.head())

# Define the portfolio performance function with dividends
def portfolio_performance(weights, total_returns):
    expected_return = np.sum(weights * total_returns.mean()) * 252  # Annualize
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(total_returns.cov() * 252, weights)))
    return expected_return, portfolio_std

# Number of portfolios to simulate
num_portfolios = 10000
results = np.zeros((3, num_portfolios))

# Simulate portfolios
for i in range(num_portfolios):
    weights = np.random.random(len(selected_tickers))
    weights /= np.sum(weights)
    expected_return, portfolio_std = portfolio_performance(weights, total_returns)
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
