import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Set up the Streamlit app layout
st.title("Portfolio Optimization Simulator")
st.write("This app allows you to optimize portfolio weights based on historical data.")

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
uploaded_file = st.file_uploader("Or upload your portfolio CSV - Tickers in the first column, weights in the second column (optional)", type=["csv"])



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
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    dividends = yf.download(tickers, start=start_date, end=end_date, actions=True, progress=False)['Dividends']
    
    # Handle any missing data by forward/backward filling
    data = data.fillna(method='ffill').fillna(method='bfill')
    dividends = dividends.fillna(0)  # Replace NaN dividends with 0

    # Calculate total dividends paid per ticker
    daily_dividends = dividends.sum(axis=0)  # Sum daily dividends for each ticker


    # Convert to annualized dividends
    # Calculate the difference in years between start and end dates
    total_days = (end_date - start_date).days
    years_difference = total_days / 365.25  # Account for leap years
    annual_dividends = daily_dividends / years_difference

    # Adjust returns to account for dividends
    dividend_yield = dividends / data  # Daily dividend yield
    log_price_returns = np.log(data / data.shift(1))  # Log price returns
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
    st.write(f"Optimized Sharpe: {optimized_sharpe:4f}")

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
    
    # Sort the portfolios by Sharpe ratio in descending order
    top_5_portfolios = portfolios_df.sort_values(by="Sharpe Ratio", ascending=False).head(5)

        # Plotting the efficient frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Return')
    plt.title('Simulated Portfolios (Efficient Frontier)')
    st.pyplot(plt.gcf())

    # Display expected return, Sharpe ratio, and standard deviation on top
    st.subheader("Top 5 Portfolios Based on Sharpe Ratio")
    portfolio_counter = 1
    for index, row in top_5_portfolios.iterrows():
        st.subheader(f"Porfolio {portfolio_counter}")
        st.write(f"Expected Return: {row['Expected Return']:.4f}")
        st.write(f"Sharpe Ratio: {row['Sharpe Ratio']:.4f}")
        st.write(f"Portfolio Standard Deviation: {row['Risk (Std Dev)']:.4f}")
        st.write("Ticker Weights:")
        portfolio_counter = portfolio_counter+1
        
        # Convert weights for this portfolio into a DataFrame with tickers and their respective weights
        weights_df = pd.DataFrame({
            "Ticker": tickers,
            "Weight": row['Weights']
        })

        st.dataframe(weights_df)
    

else:
    st.write("Please provide tickers either manually or via a CSV to proceed with optimization.")
