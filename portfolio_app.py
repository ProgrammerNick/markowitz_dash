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

# Function to validate tickers using yfinance
def validate_tickers(tickers):
    """Validate tickers by checking if they exist in Yahoo Finance"""
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        try:
            # Create a ticker object
            ticker_obj = yf.Ticker(ticker)
            # Try to get basic info - this will fail for invalid tickers
            info = ticker_obj.info
            # Check if we got meaningful data
            if info and not info.get('symbol') is None:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        except Exception:
            invalid_tickers.append(ticker)
    
    return valid_tickers, invalid_tickers

# Function to fetch data with better error handling
def fetch_stock_data(tickers, start_date, end_date):
    """Fetch stock data with improved error handling"""
    try:
        # Adjust end_date to include the specified date
        adjusted_end_date = end_date + timedelta(days=1)
        
        # Try to download data
        data = yf.download(tickers, start=start_date, end=adjusted_end_date, progress=False, auto_adjust=True)['Close']
        
        # Handle case where no data is returned
        if data.empty:
            return None, "No data returned for the specified tickers and date range."
            
        return data, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

# Proceed with optimization if tickers are provided
if tickers:
    # Validate tickers first
    st.write(f"Validating tickers: {tickers}...")
    valid_tickers, invalid_tickers = validate_tickers(tickers)
    
    # Report invalid tickers
    if invalid_tickers:
        st.error(f"The following tickers are not recognized or invalid: {', '.join(invalid_tickers)}")
        
    # Proceed only if we have valid tickers
    if not valid_tickers:
        st.error("No valid tickers found. Please check your ticker symbols and try again.")
        st.stop()
        
    # Use only valid tickers
    if len(valid_tickers) != len(tickers):
        st.info(f"Proceeding with valid tickers only: {', '.join(valid_tickers)}")
        tickers = valid_tickers
        # Update initial guess if needed
        initial_guess = np.ones(len(tickers)) / len(tickers)
    
    # Fetch adjusted close prices
    st.write(f"Fetching data for tickers: {tickers}...")
    
    data, error_message = fetch_stock_data(tickers, start_date, end_date)
    
    if error_message:
        st.error(error_message)
        st.stop()
        
    if data is None:
        st.error("Failed to fetch data for the specified tickers and date range.")
        st.stop()
    
    # Handle single ticker case
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    
    # Check if we have sufficient data
    if len(data) < 2:
        st.error("Insufficient data points for analysis. Please try a different date range.")
        st.stop()
    
    if data.isna().all().any():
        st.error("No valid data available for the specified tickers. Please check your ticker symbols.")
        st.stop()
        
    if data.isna().any().any():
        data = data.dropna()
        if len(data) < 2:
            st.error("Insufficient data points after removing missing values. Please try a different date range or tickers.")
            st.stop()
        first_date = data.index[0].strftime('%Y/%m/%d')
        last_date = data.index[-1].strftime('%Y/%m/%d')
        st.warning(f"Some stocks have missing data. Using only complete data periods. \n\nDate range being used: {first_date} to {last_date}")

    # Calculate total returns using adjusted close prices
    total_returns = np.log(data / data.shift(1)).fillna(0)
    
    # Check if we have enough data for returns calculation
    if len(total_returns) < 2:
        st.error("Insufficient data for returns calculation. Please try a different date range.")
        st.stop()

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

    # Perform optimization with error handling
    try:
        result = minimize(negative_sharpe_ratio, initial_guess, args=(total_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Check if optimization was successful
        if not result.success:
            st.warning("Optimization did not converge successfully. Results may not be optimal.")
            st.write(f"Optimization message: {result.message}")
        
        optimized_weights = result.x
        
        # Validate that we have valid weights
        if np.isnan(optimized_weights).any() or np.isinf(optimized_weights).any():
            st.error("Optimization produced invalid weights. Please try different tickers or date range.")
            st.stop()
            
        optimized_return, optimized_risk = portfolio_performance(optimized_weights, total_returns)
        
        # Validate that we have valid results
        if np.isnan(optimized_return) or np.isnan(optimized_risk) or np.isnan(result.fun):
            st.error("Optimization produced invalid results. Please try different tickers or date range.")
            st.stop()
            
        optimized_sharpe = result.fun * -1
        st.write(f"Optimized Expected Annual Return: {optimized_return:.4f}")
        st.write(f"Optimized Annual Risk (Standard Deviation): {optimized_risk:.4f}")
        st.write(f"Optimized Sharpe: {optimized_sharpe:.4f}")

        optimized_weights_df = pd.DataFrame({"Ticker": tickers, "Optimized Weight": optimized_weights})
        st.subheader("Optimized Portfolio Weights")
        st.dataframe(optimized_weights_df)
    except Exception as e:
        st.error(f"Error during portfolio optimization: {str(e)}")
        st.stop()

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

    # Check if we have valid portfolios
    if len(portfolios) == 0:
        st.error("Failed to generate portfolio simulations. Please try different tickers or date range.")
        st.stop()
        
    portfolios_df = pd.DataFrame(portfolios, columns=["Expected Return", "Risk (Std Dev)", "Sharpe Ratio", "Weights"])

    # Check if we have valid data for plotting
    if portfolios_df.empty:
        st.error("No valid portfolio data to display. Please try different tickers or date range.")
        st.stop()

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

    # Check if optimization was successful before displaying results
    if not result.success:
        st.warning("Optimization did not converge. Results may not be optimal.")
    
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