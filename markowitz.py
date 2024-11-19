import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Set up the Streamlit app layout
st.title("Portfolio Optimization Simulator")
st.write("This app allows you to simulate optimized portfolios and analyze risk-return trade-offs.")

# Input for tickers
tickers_input = st.text_input("Enter tickers separated by commas (e.g., ISTB, GLD, NVDA, MSFT, TSLA, AAPL):")
tickers = list(sorted([ticker.strip() for ticker in tickers_input.split(',')]))

# Date selection
start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-10-01"))

# Risk-free rate input
risk_free_rate = st.number_input("Risk-Free Rate", value=0.04242)

# Button to fetch data and run the simulation
if st.button("Run Simulation"):
    # Fetching data from Yahoo Finance
    if tickers:
        # Fetch adjusted close data
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        # Fetch dividend data separately for each ticker
        dividends = pd.DataFrame()
        for ticker in tickers:
            ticker_data = yf.Ticker(ticker)
            dividends[ticker] = ticker_data.dividends[start_date:end_date]  # Filter dividends for the date range
        
        # Adjusting the price data to include dividends
        # Dividend yield is calculated as Dividend / Price for each day
        dividend_yield = dividends / data

        # Calculate total return (price return + dividend yield)
        total_returns = np.log(data / data.shift(1)) + dividend_yield.shift(1)  # Adding the dividend to the log returns

        def portfolio_performance(weights, total_returns):
            expected_return = np.sum(weights * total_returns.mean()) * 252  # Annualized return
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(total_returns.cov() * 252, weights)))  # Annualized risk
            return expected_return, portfolio_std

        # Run the Monte Carlo simulation for portfolio optimization
        num_portfolios = 10000
        results = np.zeros((3, num_portfolios))
        tracked_weights = []
        tracked_returns = []

        for i in range(num_portfolios):
            # Generate random weights and normalize
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)
            tracked_weights.append(weights)

            # Calculate portfolio performance
            expected_return, portfolio_std = portfolio_performance(weights, total_returns)
            tracked_returns.append(expected_return)

            # Store the results
            results[0, i] = portfolio_std  # Risk
            results[1, i] = expected_return  # Return
            results[2, i] = (expected_return - risk_free_rate) / portfolio_std  # Sharpe ratio

        # Plotting the simulated portfolios
        st.subheader("Simulated Portfolios")
        plt.figure(figsize=(10, 6))
        plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Risk (Standard Deviation)')
        plt.ylabel('Return')
        plt.title('Simulated Portfolios')
        st.pyplot(plt.gcf())

        # Displaying top 5 portfolios with the best Sharpe ratios
        best_weights = sorted(zip(results[2], tracked_weights, tracked_returns), key=lambda x: x[0], reverse=True)[:5]
        st.subheader("Top 5 Portfolios by Sharpe Ratio")
        
        for sharpe, weights, returns in best_weights:
            st.write(f"Sharpe Ratio: {sharpe:.4f}")
            st.write(f"Expected Return: {returns:.4f}")
            weights_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
            st.dataframe(weights_df)
            st.write("---")
