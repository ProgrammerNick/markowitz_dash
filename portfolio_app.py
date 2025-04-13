import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pytz
import plotly.graph_objects as go  # Add Plotly for interactive plotting

# Set up the Streamlit app layout
st.title("Portfolio Optimization Simulator")
st.write("This app allows you to optimize portfolio weights based on historical data. To view the Github for documentation & code, click on this link [here](https://github.com/ProgrammerNick/markowitz_dash/)")

tickers = None

# Option to manually input tickers
tickers_input = st.text_input("Enter tickers separated by commas (e.g., AAPL, MSFT, GOOGL):")

# Get historical data for the tickers
start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-10-01"))

risk_free_rate = float(st.text_input("Enter risk free rate:", .04242))

if st.button("Run Simulation"):
    # If no file uploaded, use manual input
    if tickers_input:
        tickers = sorted([ticker.strip() for ticker in tickers_input.split(',')])
        initial_guess = np.ones(len(tickers)) / len(tickers)  # Default to equal weights if no file is uploaded
    else:
        st.warning("Please manually input tickers to proceed.")

# Option to upload portfolio CSV
uploaded_file = st.file_uploader("Or upload your portfolio CSV - tickers in the first column, weights in the second column (optional)", type=["csv"])

# Process input from the manual tickers or uploaded CSV
if uploaded_file is not None:
    # Read the CSV into a pandas dataframe
    portfolio_df = pd.read_csv(uploaded_file)

    # Display the uploaded CSV data
    st.write("Uploaded Portfolio (Tickers and optionally Weights):")
    st.dataframe(portfolio_df)

    # Check if 'Ticker' is in the CSV and proceed accordingly
    if 'Ticker' in portfolio_df.columns:
        tickers = portfolio_df['Ticker'].tolist()

        # If 'Weight' column exists, use it as the initial guess
        if 'Weight' in portfolio_df.columns:
            initial_guess = portfolio_df['Weight'].tolist()
            if sum(initial_guess) != 1:
                st.warning("The sum of the weights in the uploaded CSV doesn't equal 1. We will normalize them.")
                initial_guess = np.array(initial_guess) / sum(initial_guess)  # Normalize to sum to 1
        else:
            # Default to equal weights if no weights are provided
            initial_guess = np.ones(len(tickers)) / len(tickers)

# Proceed with optimization if tickers are provided
if tickers:
    # Fetch adjusted close prices for tickers
    st.write(f"Fetching data for tickers: {tickers}...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    
    # Handle case where data is a Series (single ticker) by converting to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    
    # Handle any missing data by forward/backward filling
    data = data.ffill().bfill()

    # Initialize dividends DataFrame
    dividends = pd.DataFrame(index=data.index, columns=data.columns).fillna(0)

    # Convert start_date and end_date to timezone-aware datetime
    tz = pytz.timezone('America/New_York')
    start_date = pd.to_datetime(start_date).normalize().tz_localize(None)
    end_date = pd.to_datetime(end_date).normalize().tz_localize(None)

    # Fetch dividends individually for each ticker
    st.write("Fetching dividend data...")
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            div = ticker_obj.dividends
            div.index = pd.to_datetime(div.index).normalize().tz_localize(None)
            # Filter dividends within the date range
            div = div.loc[(div.index >= start_date) & (div.index <= end_date)]
            # Resample to match data index (daily), forward fill, and align
            div = div.reindex(data.index).fillna(0)
            dividends[ticker] = div
        except Exception as e:
            st.warning(f"Could not fetch dividends for {ticker}: {e}. Assuming zero dividends.")
            dividends[ticker] = 0

    # Calculate total dividends paid per ticker
    daily_dividends = dividends.sum(axis=0)  # Sum daily dividends for each ticker

    # Convert to annualized dividends
    total_days = (end_date - start_date).days
    years_difference = total_days / 365.25  # Account for leap years
    annual_dividends = daily_dividends / years_difference

    # Adjust returns to account for dividends
    dividend_yield = dividends / data  # Daily dividend yield
    dividend_yield = dividend_yield.fillna(0)  # Replace NaN with 0
    log_price_returns = np.log(data / data.shift(1))  # Log price returns
    log_price_returns = log_price_returns.fillna(0)  # Handle NaN from first row
    total_returns = log_price_returns + dividend_yield.shift(1).fillna(0)  # Add dividend yield to total returns

    # Display the annualized dividends
    st.subheader("Annualized Dividends per Ticker")
    dividends_df = pd.DataFrame({
        "Annualized Dividends": annual_dividends
    })
    st.dataframe(dividends_df)

    def portfolio_performance(weights, total_returns):
        expected_return = np.sum(weights * total_returns.mean()) * 252  # Annualized return
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(total_returns.cov() * 252, weights)))  # Annualized risk
        return expected_return, portfolio_std

    # Optimization function (maximize Sharpe ratio)
    def negative_sharpe_ratio(weights, total_returns, risk_free_rate=0.04242):
        expected_return, portfolio_std = portfolio_performance(weights, total_returns)
        return -(expected_return - risk_free_rate) / portfolio_std  # Return negative Sharpe to minimize

    # Run optimization to find the optimal weights
    num_assets = len(tickers)
    bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Weights must sum to 1

    result = minimize(negative_sharpe_ratio, initial_guess, args=(total_returns,), method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the optimized weights
    optimized_weights = result.x

    # Portfolio performance
    optimized_return, optimized_risk = portfolio_performance(optimized_weights, total_returns)
    optimized_sharpe = result.fun * -1
    st.write(f"Optimized Expected Annual Return: {optimized_return:.4f}")
    st.write(f"Optimized Annual Risk (Standard Deviation): {optimized_risk:.4f}")
    st.write(f"Optimized Sharpe: {optimized_sharpe:.4f}")

    optimized_weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Optimized Weight": optimized_weights
    })

    st.subheader("Optimized Portfolio Weights")
    st.dataframe(optimized_weights_df)

    # Plotting the efficient frontier
    st.subheader("Efficient Frontier")
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    portfolios = []
    for i in range(num_portfolios):
        # Generate random weights and normalize
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        expected_return, portfolio_std = portfolio_performance(weights, total_returns)
        results[0, i] = portfolio_std
        results[1, i] = expected_return
        results[2, i] = (expected_return - risk_free_rate) / portfolio_std  # Sharpe ratio

        # Store the portfolios and their weights
        portfolios.append([expected_return, portfolio_std, results[2, i], weights])

    # Convert portfolios to a DataFrame
    portfolios_df = pd.DataFrame(portfolios, columns=["Expected Return", "Risk (Std Dev)", "Sharpe Ratio", "Weights"])

    # Create hover text with portfolio weights
    hover_texts = []
    for i, row in portfolios_df.iterrows():
        weights = row['Weights']
        # Format weights as a string for each ticker
        weights_str = '<br>'.join([f"{ticker}: {weight:.2%}" for ticker, weight in zip(tickers, weights)])
        hover_text = (
            f"Return: {row['Expected Return']:.2%}<br>"
            f"Risk: {row['Risk (Std Dev)']:.2%}<br>"
            f"Sharpe Ratio: {row['Sharpe Ratio']:.4f}<br>"
            f"Weights:<br>{weights_str}"
        )
        hover_texts.append(hover_text)

    # Create Plotly scatter plot
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

    # Update layout
    fig.update_layout(
        title="Simulated Portfolios (Efficient Frontier)",
        xaxis_title="Risk (Standard Deviation)",
        yaxis_title="Return",
        width=800,
        height=600,
        showlegend=False,
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Sort the portfolios by Sharpe ratio in descending order
    top_5_portfolios = portfolios_df.sort_values(by="Sharpe Ratio", ascending=False).head(5)

    # Display expected return, Sharpe ratio, and standard deviation on top
    st.subheader("Top 5 Portfolios Based on Sharpe Ratio")
    portfolio_counter = 1
    for index, row in top_5_portfolios.iterrows():
        st.subheader(f"Portfolio {portfolio_counter}")
        st.write(f"Expected Return: {row['Expected Return']:.4f}")
        st.write(f"Sharpe Ratio: {row['Sharpe Ratio']:.4f}")
        st.write(f"Portfolio Standard Deviation: {row['Risk (Std Dev)']:.4f}")
        st.write("Ticker Weights:")
        portfolio_counter += 1
        
        # Convert weights for this portfolio into a DataFrame with tickers and their respective weights
        weights_df = pd.DataFrame({
            "Ticker": tickers,
            "Weight": row['Weights']
        })

        st.dataframe(weights_df)
    
else:
    st.write("Please provide tickers either manually or via a CSV to proceed with optimization.")