import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

# Set up the Streamlit app layout with page config
st.set_page_config(
    page_title="Portfolio Optimization Simulator",
    layout="wide"
)

# Custom CSS for Mobile Responsiveness, Typography, and UI Components
st.markdown("""
<style>
    /* Mobile responsive container padding */
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1.5rem !important;
        max-width: 100% !important;
    }
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #1F2937;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 0.5rem;
    }
    /* Mobile Media Queries */
    @media (max-width: 768px) {
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Restored initial exact title and description
st.title("Portfolio Optimization Simulator")
st.write("This app allows you to optimize portfolio weights based on historical data by utilizing Modern Portfolio Theory. To view the Github for documentation & code, click on this link [here](https://github.com/ProgrammerNick/markowitz_dash/)")

# Initialize session state for tickers and initial guess if not set
if "tickers" not in st.session_state:
    st.session_state.tickers = None
if "initial_guess" not in st.session_state:
    st.session_state.initial_guess = None

# Input section organized inside a styled container
with st.expander("Simulation Parameters & Input Configuration", expanded=True):
    col_input1, col_input2 = st.columns([2, 1])
    
    with col_input1:
        tickers_input = st.text_input(
            "Enter Ticker Symbols (comma-separated):",
            value="AAPL, MSFT, GOOGL",
            help="Enter valid stock/ETF tickers separated by commas, e.g. AAPL, MSFT, NVDA, SPY"
        )
    
    with col_input2:
        uploaded_file = st.file_uploader(
            "Or Upload Portfolio CSV (optional)",
            type=["csv"],
            help="CSV with 'Ticker' column and optional 'Weight' column"
        )

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
            "Risk-Free Rate (Annual Decimal):",
            value=0.04242,
            format="%.5f",
            step=0.001,
            help="Current benchmark risk-free rate, e.g. 0.0424 for 4.24%"
        )

    if start_date >= end_date:
        st.error("Start Date must be strictly before End Date.")
        st.stop()

    run_button = st.button("Run Portfolio Optimization", use_container_width=True)

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
    with st.spinner(f"Fetching historical price data for {', '.join(tickers)}..."):
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

    if len(data) < 2:
        st.error("Insufficient data points for analysis. Please try a wider date range.")
        st.stop()
        
    if data.isna().any().any():
        data = data.dropna()
        if len(data) < 2:
            st.error("Insufficient overlapping data points after removing missing values. Please try a different date range or tickers.")
            st.stop()

    # Calculate simple percentage returns (standard for linear portfolio aggregation Rp = sum(w_i * R_i))
    returns = data.pct_change().dropna()
    
    if len(returns) < 2:
        st.error("Insufficient data for returns calculation. Please try a different date range.")
        st.stop()

    def portfolio_performance(weights, returns_df):
        expected_return = np.sum(weights * returns_df.mean()) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * 252, weights)))
        return expected_return, portfolio_std

    def negative_sharpe_ratio(weights, returns_df, rf_rate):
        expected_return, portfolio_std = portfolio_performance(weights, returns_df)
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

    # Perform optimization
    try:
        result = minimize(
            negative_sharpe_ratio,
            initial_guess,
            args=(returns, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            st.warning("Optimization did not converge successfully. Results may not be optimal.")
        
        optimized_weights = result.x
        
        if np.isnan(optimized_weights).any() or np.isinf(optimized_weights).any():
            st.error("Optimization produced invalid weights. Please try different tickers or date range.")
            st.stop()
            
        optimized_return, optimized_risk = portfolio_performance(optimized_weights, returns)
        
        if np.isnan(optimized_return) or np.isnan(optimized_risk) or np.isnan(result.fun):
            st.error("Optimization produced invalid results. Please try different tickers or date range.")
            st.stop()
            
        optimized_sharpe = -result.fun
        
    except Exception as e:
        st.error(f"Error during portfolio optimization: {str(e)}")
        st.stop()

    # Efficient frontier Monte Carlo simulation with Dirichlet uniform simplex sampling
    num_portfolios = 10000
    portfolios = []
    np.random.seed(42)
    for i in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(num_assets))
        expected_return, portfolio_std = portfolio_performance(weights, returns)
        sharpe = (expected_return - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
        portfolios.append([expected_return, portfolio_std, sharpe, weights])

    if len(portfolios) == 0:
        st.error("Failed to generate portfolio simulations. Please try different tickers or date range.")
        st.stop()
        
    portfolios_df = pd.DataFrame(portfolios, columns=["Expected Return", "Risk (Std Dev)", "Sharpe Ratio", "Weights"])

    # Calculate individual asset return and std for chart overlay
    asset_stats = []
    for t in tickers:
        ret = returns[t].mean() * 252
        std = returns[t].std() * np.sqrt(252)
        sharpe = (ret - risk_free_rate) / std if std > 0 else 0
        asset_stats.append({'Ticker': t, 'Return': ret, 'Risk': std, 'Sharpe': sharpe})
    asset_df = pd.DataFrame(asset_stats)

    # Strategic Portfolios Calculations
    min_var_idx = portfolios_df["Risk (Std Dev)"].idxmin()
    min_var_row = portfolios_df.loc[min_var_idx]
    
    eq_w = np.ones(num_assets) / num_assets
    eq_ret, eq_std = portfolio_performance(eq_w, returns)
    eq_sharpe = (eq_ret - risk_free_rate) / eq_std if eq_std > 0 else 0
    
    max_ret_idx = portfolios_df["Expected Return"].idxmax()
    max_ret_row = portfolios_df.loc[max_ret_idx]

    # Filter top 5 distinct portfolios by allocation distance
    df_sorted = portfolios_df.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)
    distinct_top5 = []
    for idx, row in df_sorted.iterrows():
        w = row['Weights']
        if not distinct_top5:
            distinct_top5.append(row)
        else:
            is_distinct = True
            for prev in distinct_top5:
                if np.max(np.abs(w - prev['Weights'])) < 0.05:
                    is_distinct = False
                    break
            if is_distinct:
                distinct_top5.append(row)
        if len(distinct_top5) == 5:
            break

    distinct_df = pd.DataFrame(distinct_top5)

    # --- TOP LEVEL SUMMARY METRICS ---
    st.markdown("---")
    st.subheader("Maximum Sharpe Ratio Portfolio (SLSQP Optimal)")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Optimized Expected Return", f"{optimized_return:.2%}")
    col_m2.metric("Optimized Annual Risk (Std Dev)", f"{optimized_risk:.2%}")
    col_m3.metric("Optimized Sharpe Ratio", f"{optimized_sharpe:.4f}")
    col_m4.metric("Assets Analyzed", f"{len(tickers)}")

    # --- EFFICIENT FRONTIER SCATTER PLOT DISPLAYED DIRECTLY ---
    st.markdown("---")
    st.subheader("Simulated Portfolios & Efficient Frontier")
    
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
    
    # Trace 1: 10,000 Simulated Portfolios
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
                colorbar=dict(
                    title="Sharpe",
                    thickness=12,
                    len=0.75
                ),
                opacity=0.75
            ),
            text=hover_texts,
            hoverinfo='text',
            name="10,000 Portfolios"
        )
    )

    # Trace 2: Individual Assets Markers with mobile-optimized centered labels & cliponaxis=False
    fig.add_trace(
        go.Scatter(
            x=asset_df['Risk'],
            y=asset_df['Return'],
            mode='markers+text',
            marker=dict(size=12, color='#DC2626', line=dict(width=2, color='black')),
            text=[f"<b>{t}</b>" for t in asset_df['Ticker']],
            textposition="top center",
            textfont=dict(size=12, color="#1F2937", family="Arial, sans-serif"),
            cliponaxis=False,
            name="Individual Stocks",
            hoverinfo='text',
            hovertext=[f"Stock: {row['Ticker']}<br>Return: {row['Return']:.2%}<br>Risk: {row['Risk']:.2%}<br>Sharpe: {row['Sharpe']:.4f}" for _, row in asset_df.iterrows()]
        )
    )

    fig.update_layout(
        xaxis=dict(
            title="Annualized Risk (Standard Deviation)",
            tickformat=".1%",
            gridcolor="#F3F4F6"
        ),
        yaxis=dict(
            title="Annualized Expected Return",
            tickformat=".1%",
            gridcolor="#F3F4F6"
        ),
        template="plotly_white",
        autosize=True,
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=35, b=35),
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)

    col_df1, col_df2 = st.columns([1, 1])
    with col_df1:
        st.subheader("Optimal Asset Weights (Max Sharpe Ratio)")
        optimized_weights_df = pd.DataFrame({
            "Ticker": tickers,
            "Optimized Weight": [f"{w:.2%}" for w in optimized_weights]
        })
        st.dataframe(optimized_weights_df, use_container_width=True)
        
    with col_df2:
        st.subheader("Individual Asset Statistics")
        asset_summary_df = pd.DataFrame({
            "Ticker": asset_df["Ticker"],
            "Annual Return": [f"{r:.2%}" for r in asset_df["Return"]],
            "Annual Risk": [f"{s:.2%}" for s in asset_df["Risk"]],
            "Sharpe Ratio": [f"{sh:.4f}" for sh in asset_df["Sharpe"]]
        })
        st.dataframe(asset_summary_df, use_container_width=True)

    # --- STRATEGIC BENCHMARKS & TOP PORTFOLIOS SECTION ---
    st.markdown("---")
    st.subheader("Strategic Benchmark Portfolios Comparison")
    
    benchmark_summary = pd.DataFrame([
        {
            "Strategy": "Maximum Sharpe Ratio (SLSQP Optimal)",
            "Expected Return": f"{optimized_return:.2%}",
            "Risk (Std Dev)": f"{optimized_risk:.2%}",
            "Sharpe Ratio": f"{optimized_sharpe:.4f}",
            "Allocation Breakdown": ", ".join([f"{t}: {w:.1%}" for t, w in zip(tickers, optimized_weights)])
        },
        {
            "Strategy": "Minimum Volatility (Lowest Risk)",
            "Expected Return": f"{min_var_row['Expected Return']:.2%}",
            "Risk (Std Dev)": f"{min_var_row['Risk (Std Dev)']:.2%}",
            "Sharpe Ratio": f"{min_var_row['Sharpe Ratio']:.4f}",
            "Allocation Breakdown": ", ".join([f"{t}: {w:.1%}" for t, w in zip(tickers, min_var_row['Weights'])])
        },
        {
            "Strategy": "Equal Weight (1/N Benchmark)",
            "Expected Return": f"{eq_ret:.2%}",
            "Risk (Std Dev)": f"{eq_std:.2%}",
            "Sharpe Ratio": f"{eq_sharpe:.4f}",
            "Allocation Breakdown": ", ".join([f"{t}: {w:.1%}" for t, w in zip(tickers, eq_w)])
        },
        {
            "Strategy": "Maximum Return Focus",
            "Expected Return": f"{max_ret_row['Expected Return']:.2%}",
            "Risk (Std Dev)": f"{max_ret_row['Risk (Std Dev)']:.2%}",
            "Sharpe Ratio": f"{max_ret_row['Sharpe Ratio']:.4f}",
            "Allocation Breakdown": ", ".join([f"{t}: {w:.1%}" for t, w in zip(tickers, max_ret_row['Weights'])])
        }
    ])
    
    st.dataframe(benchmark_summary, use_container_width=True)

    st.subheader("Top 5 High-Sharpe Portfolios")
    portfolio_counter = 1
    for index, row in distinct_df.reset_index(drop=True).iterrows():
        with st.expander(f"Portfolio {portfolio_counter} — Sharpe Ratio: {row['Sharpe Ratio']:.4f}", expanded=(portfolio_counter == 1)):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Expected Return:** {row['Expected Return']:.2%}")
            c2.write(f"**Risk (Std Dev):** {row['Risk (Std Dev)']:.2%}")
            c3.write(f"**Sharpe Ratio:** {row['Sharpe Ratio']:.4f}")
            weights_df = pd.DataFrame({"Ticker": tickers, "Weight": [f"{w:.2%}" for w in row['Weights']]})
            st.dataframe(weights_df, use_container_width=True)
        portfolio_counter += 1

    with st.expander("Historical Price & Returns Data Preview"):
        st.markdown("### Historical Adjusted Close Price Data")
        st.dataframe(data, use_container_width=True)
        st.markdown("### Daily Percentage Returns")
        st.dataframe(returns, use_container_width=True)

else:
    st.info("Please enter tickers and click 'Run Portfolio Optimization' or upload a portfolio CSV to display optimization data.")
