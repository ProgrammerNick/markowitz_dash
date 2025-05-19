import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

# Set up the Streamlit app layout
st.title("Portfolio Optimization Simulator")
st.write("This app allows you to optimize portfolio weights based on historical data by utilizing Modern Portfolio Theory. To view the Github for documentation & code, click on this link [here](https://github.com/ProgrammerNick/markowitz_dash/)")

tickers = None

# Option to manually input tickers
tickers_input = st.text_input("Enter tickers separated by commas (e.g., AAPL, MSFT, GOOGL):")
max_date = date.today()

# Set up date inputs with min and max values
start_date = st.date_input(
    "Start Date",
    value=pd.to_datetime("2021-01-01"),
    min_value=pd.to_datetime("1962-01-01"),
    max_value=max_date
)
end_date = st.date_input(
    "End Date",
    value=pd.to_datetime("2024-10-01"),
    min_value=pd.to_datetime("1962-01-01"),
    max_value=max_date
)

risk_free_rate = float(st.text_input("Enter risk free rate:", .04242))

if st.button("Run Simulation"):
    if tickers_input:
        tickers = sorted([ticker.strip() for ticker in tickers_input.split(',')])
        initial_guess = np.ones(len(tickers)) / len(tickers)
    else:
        st.warning("Please manually input tickers to proceed.")

# Option to upload portfolio CSV
uploaded_file = st.file_uploader("Or upload your portfolio CSV - tickers in the first column, weights in the second column (optional)", type=["csv"])

# Process input from the manual tickers or uploaded CSV
if uploaded_file is not None:
    portfolio_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Portfolio (Tickers and optionally Weights):")
    st.dataframe(portfolio_df)
    if 'Ticker' in portfolio_df.columns:
        tickers = portfolio_df['Ticker'].tolist()
        if 'Weight' in portfolio_df.columns:
            initial_guess = portfolio_df['Weight'].tolist()
            if sum(initial_guess) != 1:
                st.warning("Weights don't sum to 1. Normalizing...")
                initial_guess = np.array(initial_guess) / sum(initial_guess)
        else:
            initial_guess = np.ones(len(tickers)) / len(tickers)

# Proceed with optimization if tickers are provided
if tickers:
    # Fetch adjusted close prices
    st.write(f"Fetching data for tickers: {tickers}...")

    # Assuming start_date and end_date are from st.date_input
    # Adjust end_date to include the specified date
    adjusted_end_date = end_date + timedelta(days=1)

    data = yf.download(tickers, start=start_date, end=adjusted_end_date, progress=False, auto_adjust=True)['Close']
    
    # Handle single ticker case
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    
    if data.isna().any().any():
        data = data.dropna()
        first_date = data.index[0].strftime('%Y/%m/%d')
        last_date = data.index[-1].strftime('%Y/%m/%d')
        st.warning(f"Some stocks have missing data. Using only complete data periods. \n\nData range being used: {first_date} to {last_date}")

    # Calculate total returns using adjusted close prices
    total_returns = np.log(data / data.shift(1)).fillna(0)

    def portfolio_performance(weights, total_returns):
        expected_return = np.sum(weights * total_returns.mean()) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(total_returns.cov() * 252, weights)))
        return expected_return, portfolio_std

    def negative_sharpe_ratio(weights, total_returns, risk_free_rate=0.04242):
        expected_return, portfolio_std = portfolio_performance(weights, total_returns)
        return -(expected_return - risk_free_rate) / portfolio_std

    num_assets = len(tickers)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    result = minimize(negative_sharpe_ratio, initial_guess, args=(total_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_weights = result.x

    optimized_return, optimized_risk = portfolio_performance(optimized_weights, total_returns)
    optimized_sharpe = result.fun * -1
    st.write(f"Optimized Expected Annual Return: {optimized_return:.4f}")
    st.write(f"Optimized Annual Risk (Standard Deviation): {optimized_risk:.4f}")
    st.write(f"Optimized Sharpe: {optimized_sharpe:.4f}")

    optimized_weights_df = pd.DataFrame({"Ticker": tickers, "Optimized Weight": optimized_weights})
    st.subheader("Optimized Portfolio Weights")
    st.dataframe(optimized_weights_df)

    # Efficient frontier
    st.subheader("Efficient Frontier")
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    portfolios = []
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        expected_return, portfolio_std = portfolio_performance(weights, total_returns)
        results[0, i] = portfolio_std
        results[1, i] = expected_return
        results[2, i] = (expected_return - risk_free_rate) / portfolio_std
        portfolios.append([expected_return, portfolio_std, results[2, i], weights])

    portfolios_df = pd.DataFrame(portfolios, columns=["Expected Return", "Risk (Std Dev)", "Sharpe Ratio", "Weights"])

    hover_texts = []
    for i, row in portfolios_df.iterrows():
        weights = row['Weights']
        weights_str = '<br>'.join([f"{ticker}: {weight:.2%}" for ticker, weight in zip(tickers, weights)])
        hover_text = (
            f"Return: {row['Expected Return']:.2%}<br>"
            f"Risk: {row['Risk (Std Dev)']:.2%}<br>"
            f"Sharpe Ratio: {row['Sharpe Ratio']:.4f}<br>"
            f"Weights:<br>{weights_str}"
        )
        hover_texts.append(hover_text)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolios_df["Risk (Std Dev)"],
            y=portfolios_df["Expected Return"],
            mode='markers',
            marker=dict(
                size=5,
                color=portfolios_df["Sharpe Ratio"],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
            ),
            text=hover_texts,
            hoverinfo='text',
        )
    )

    fig.update_layout(
        title="Simulated Portfolios (Efficient Frontier)",
        xaxis_title="Risk (Standard Deviation)",
        yaxis_title="Return",
        width=800,
        height=600,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    top_5_portfolios = portfolios_df.sort_values(by="Sharpe Ratio", ascending=False).head(5)

    st.subheader("Top 5 Portfolios Based on Sharpe Ratio")
    portfolio_counter = 1
    for index, row in top_5_portfolios.iterrows():
        st.subheader(f"Portfolio {portfolio_counter}")
        st.write(f"Expected Return: {row['Expected Return']:.4f}")
        st.write(f"Sharpe Ratio: {row['Sharpe Ratio']:.4f}")
        st.write(f"Portfolio Standard Deviation: {row['Risk (Std Dev)']:.4f}")
        st.write("Ticker Weights:")
        weights_df = pd.DataFrame({"Ticker": tickers, "Weight": row['Weights']})
        st.dataframe(weights_df)
        portfolio_counter += 1
    
else:
    st.write("Please provide tickers either manually or via a CSV to proceed with optimization.")