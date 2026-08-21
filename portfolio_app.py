import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

# Set up the Streamlit app layout
st.set_page_config(page_title="Portfolio Optimization Simulator", layout="wide")

st.title("Portfolio Optimization Simulator")
st.write("This app allows you to optimize portfolio weights based on historical data by utilizing Modern Portfolio Theory. To view the Github for documentation & code, click on this link [here](https://github.com/ProgrammerNick/markowitz_dash/)")

# Initialize session state for tickers and initial guess if not set
if "tickers" not in st.session_state:
    st.session_state.tickers = None
if "initial_guess" not in st.session_state:
    st.session_state.initial_guess = None

# Option to manually input tickers
tickers_input = st.text_input("Enter tickers separated by commas (e.g., AAPL, MSFT, GOOGL):", value="AAPL, MSFT, GOOGL")
max_date = date.today()

col1, col2, col3 = st.columns(3)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=pd.to_datetime("2021-01-01").date(),
        min_value=pd.to_datetime("1962-01-01").date(),
        max_value=max_date
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=pd.to_datetime("2024-10-01").date(),
        min_value=pd.to_datetime("1962-01-01").date(),
        max_value=max_date
    )
with col3:
    risk_free_rate = st.number_input(
        "Enter risk free rate (annual decimal):",
        value=0.04242,
        format="%.5f",
        step=0.001
    )

if start_date >= end_date:
    st.error("Start Date must be strictly before End Date.")
    st.stop()

run_button = st.button("Run Simulation")

# Option to upload portfolio CSV
uploaded_file = st.file_uploader("Or upload your portfolio CSV - tickers in the first column, weights in the second column (optional)", type=["csv"])

# Process CSV Upload
if uploaded_file is not None:
    try:
        portfolio_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Portfolio (Tickers and optionally Weights):")
        st.dataframe(portfolio_df)
        if 'Ticker' in portfolio_df.columns:
            parsed_tickers = [str(t).strip().upper() for t in portfolio_df['Ticker'].tolist() if pd.notna(t) and str(t).strip()]
            if parsed_tickers:
                st.session_state.tickers = parsed_tickers
                if 'Weight' in portfolio_df.columns:
                    weights = portfolio_df['Weight'].tolist()
                    if sum(weights) != 1 and sum(weights) > 0:
                        st.warning("Weights don't sum to 1. Normalizing...")
                        st.session_state.initial_guess = np.array(weights) / sum(weights)
                    else:
                        st.session_state.initial_guess = np.array(weights)
                else:
                    st.session_state.initial_guess = np.ones(len(parsed_tickers)) / len(parsed_tickers)
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")

# Process manual button click
if run_button:
    if tickers_input:
        parsed_tickers = sorted(list(set([t.strip().upper() for t in tickers_input.split(',') if t.strip()])))
        if parsed_tickers:
            st.session_state.tickers = parsed_tickers
            st.session_state.initial_guess = np.ones(len(parsed_tickers)) / len(parsed_tickers)
        else:
            st.warning("Please enter valid ticker symbols.")
    else:
        st.warning("Please manually input tickers to proceed.")

def fetch_stock_data(tickers, start_date, end_date):
    """Fetch stock data using yfinance with robust MultiIndex and error handling"""
    try:
        adjusted_end_date = end_date + timedelta(days=1)
        raw_data = yf.download(tickers, start=start_date, end=adjusted_end_date, progress=False, auto_adjust=True)
        
        if raw_data is None or raw_data.empty:
            return None, [], tickers, "No data returned for the specified tickers and date range."

        # Extract Close price data handling MultiIndex vs SingleIndex
        if isinstance(raw_data.columns, pd.MultiIndex):
            if 'Close' in raw_data.columns.levels[0]:
                data = raw_data['Close'].copy()
            else:
                data = raw_data.xs(raw_data.columns.levels[0][0], axis=1, level=0)
        else:
            if 'Close' in raw_data.columns:
                data = raw_data[['Close']].copy()
                data.columns = tickers[:1]
            else:
                data = raw_data.copy()

        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])

        valid_tickers = []
        invalid_tickers = []

        for col in data.columns:
            if data[col].dropna().empty:
                invalid_tickers.append(str(col))
            else:
                valid_tickers.append(str(col))

        for t in tickers:
            if t not in data.columns and t not in invalid_tickers:
                invalid_tickers.append(str(t))

        if not valid_tickers:
            return None, [], invalid_tickers, "None of the requested tickers returned price data."

        data = data[valid_tickers]
        return data, valid_tickers, invalid_tickers, None
    except Exception as e:
        return None, [], tickers, f"Error fetching data: {str(e)}"

tickers = st.session_state.tickers
initial_guess = st.session_state.initial_guess

# Proceed with optimization if tickers are provided
if tickers:
    st.write(f"Fetching historical price data for tickers: {', '.join(tickers)}...")
    
    data, valid_tickers, invalid_tickers, error_message = fetch_stock_data(tickers, start_date, end_date)
    
    if invalid_tickers:
        st.warning(f"The following tickers were not recognized or had no price data: {', '.join(invalid_tickers)}")

    if error_message:
        st.error(error_message)
        st.stop()
        
    if data is None or data.empty:
        st.error("Failed to fetch data for the specified tickers and date range.")
        st.stop()

    if len(valid_tickers) < 2:
        st.error("Modern Portfolio Theory requires at least 2 valid tickers with historical price data to optimize portfolio weights.")
        st.stop()
        
    if len(valid_tickers) != len(tickers):
        st.info(f"Proceeding with valid tickers only: {', '.join(valid_tickers)}")
        tickers = valid_tickers
        initial_guess = np.ones(len(tickers)) / len(tickers)

    # Check if we have sufficient data points
    if len(data) < 2:
        st.error("Insufficient data points for analysis. Please try a wider date range.")
        st.stop()
        
    if data.isna().any().any():
        data = data.dropna()
        if len(data) < 2:
            st.error("Insufficient overlapping data points after removing missing values. Please try a different date range or tickers.")
            st.stop()
        first_date = data.index[0].strftime('%Y/%m/%d')
        last_date = data.index[-1].strftime('%Y/%m/%d')
        st.warning(f"Some stocks have missing data. Using only complete data periods.\n\nDate range being used: {first_date} to {last_date}")

    # Calculate total returns using log returns
    total_returns = np.log(data / data.shift(1)).dropna()
    
    if len(total_returns) < 2:
        st.error("Insufficient data for returns calculation. Please try a different date range.")
        st.stop()

    def portfolio_performance(weights, total_returns):
        expected_return = np.sum(weights * total_returns.mean()) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(total_returns.cov() * 252, weights)))
        return expected_return, portfolio_std

    def negative_sharpe_ratio(weights, total_returns, rf_rate):
        expected_return, portfolio_std = portfolio_performance(weights, total_returns)
        if portfolio_std == 0 or np.isnan(portfolio_std):
            return 1e9
        return -(expected_return - rf_rate) / portfolio_std

    num_assets = len(tickers)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    if initial_guess is None or len(initial_guess) != num_assets:
        initial_guess = np.ones(num_assets) / num_assets
    else:
        initial_guess = np.array(initial_guess) / np.sum(initial_guess)

    # Perform optimization with error handling
    try:
        result = minimize(
            negative_sharpe_ratio,
            initial_guess,
            args=(total_returns, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            st.warning("Optimization did not converge successfully. Results may not be optimal.")
            st.write(f"Optimization message: {result.message}")
        
        optimized_weights = result.x
        
        if np.isnan(optimized_weights).any() or np.isinf(optimized_weights).any():
            st.error("Optimization produced invalid weights. Please try different tickers or date range.")
            st.stop()
            
        optimized_return, optimized_risk = portfolio_performance(optimized_weights, total_returns)
        
        if np.isnan(optimized_return) or np.isnan(optimized_risk) or np.isnan(result.fun):
            st.error("Optimization produced invalid results. Please try different tickers or date range.")
            st.stop()
            
        optimized_sharpe = -result.fun
        
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Optimized Annual Return", f"{optimized_return:.2%}")
        col_res2.metric("Optimized Annual Risk (Std Dev)", f"{optimized_risk:.2%}")
        col_res3.metric("Optimized Sharpe Ratio", f"{optimized_sharpe:.4f}")

        optimized_weights_df = pd.DataFrame({"Ticker": tickers, "Optimized Weight": [f"{w:.2%}" for w in optimized_weights]})
        st.subheader("Optimized Portfolio Weights")
        st.dataframe(optimized_weights_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error during portfolio optimization: {str(e)}")
        st.stop()

    # Efficient frontier simulation
    st.subheader("Efficient Frontier")
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    portfolios = []
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        expected_return, portfolio_std = portfolio_performance(weights, total_returns)
        sharpe = (expected_return - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
        results[0, i] = portfolio_std
        results[1, i] = expected_return
        results[2, i] = sharpe
        portfolios.append([expected_return, portfolio_std, sharpe, weights])

    if len(portfolios) == 0:
        st.error("Failed to generate portfolio simulations. Please try different tickers or date range.")
        st.stop()
        
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
            name="Simulated Portfolios"
        )
    )

    # Highlight optimal portfolio point
    fig.add_trace(
        go.Scatter(
            x=[optimized_risk],
            y=[optimized_return],
            mode='markers',
            marker=dict(
                size=14,
                color='red',
                symbol='star',
                line=dict(width=2, color='black')
            ),
            name="Maximum Sharpe Ratio",
            text=[f"Optimal Portfolio<br>Return: {optimized_return:.2%}<br>Risk: {optimized_risk:.2%}<br>Sharpe: {optimized_sharpe:.4f}"],
            hoverinfo='text'
        )
    )

    fig.update_layout(
        title="Simulated Portfolios (Efficient Frontier)",
        xaxis_title="Risk (Standard Deviation)",
        yaxis_title="Return",
        width=800,
        height=600,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)

    top_5_portfolios = portfolios_df.sort_values(by="Sharpe Ratio", ascending=False).head(5)

    st.subheader("Top 5 Portfolios Based on Sharpe Ratio")
    portfolio_counter = 1
    for index, row in top_5_portfolios.iterrows():
        with st.expander(f"Portfolio {portfolio_counter} (Sharpe: {row['Sharpe Ratio']:.4f})", expanded=(portfolio_counter == 1)):
            st.write(f"**Expected Return:** {row['Expected Return']:.2%}")
            st.write(f"**Risk (Std Dev):** {row['Risk (Std Dev)']:.2%}")
            st.write(f"**Sharpe Ratio:** {row['Sharpe Ratio']:.4f}")
            weights_df = pd.DataFrame({"Ticker": tickers, "Weight": [f"{w:.2%}" for w in row['Weights']]})
            st.dataframe(weights_df, use_container_width=True)
        portfolio_counter += 1
else:
    st.info("Please enter tickers and click 'Run Simulation' or upload a portfolio CSV to display optimization data.")
