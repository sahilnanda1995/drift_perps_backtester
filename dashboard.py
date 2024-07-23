import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta, date

# Set page config
st.set_page_config(page_title="Drift Protocol Fee Analysis", layout="wide")

# Title
st.title("Drift Protocol Fee and Profit Analysis")

# Load funding rate data
@st.cache_data
def load_funding_rates(token):
    try:
        df = pd.read_csv(f"{token.lower()}.csv", sep=";")
        df['ts'] = pd.to_datetime(df['ts'], format="%m/%d/%Y, %H:%M:%S")
        return df
    except Exception as e:
        st.error(f"Error loading funding rates for {token}: {e}")
        return pd.DataFrame()

# Fetch price data
@st.cache_data
def fetch_price_data(token, start_date, end_date):
    try:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        url = f"https://benchmarks.pyth.network/v1/shims/tradingview/history"
        params = {
            "symbol": f"Crypto.{token}/USD",
            "resolution": "60",
            "from": int(start_datetime.timestamp()),
            "to": int(end_datetime.timestamp())
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['t'], unit='s'),
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c']
        })
        return df
    except requests.RequestException as e:
        st.error(f"Error fetching price data: {e}")
        return pd.DataFrame()
    except (KeyError, ValueError) as e:
        st.error(f"Error processing price data: {e}")
        return pd.DataFrame()

# Preprocess funding rate data
@st.cache_data
def preprocess_funding_rates(funding_rates):
    return funding_rates.sort_values('ts')

# Map funding rates to price data
@st.cache_data
def map_funding_rates_to_price(funding_rates, price_data):
    return pd.merge_asof(price_data, funding_rates, left_on='timestamp', right_on='ts', direction='nearest')

# Calculate fees and profits
def calculate_fees_and_profits(mapped_data, entry_amount, leverage, margin_direction, token):
    """
    Calculate fees and profits for a given trading scenario.

    Args:
    mapped_data (pd.DataFrame): DataFrame containing price and funding rate data
    entry_amount (float): Initial investment amount in USD
    leverage (float): Leverage used for the trade
    margin_direction (str): 'Long' or 'Short'
    token (str): Cryptocurrency token symbol

    Returns:
    tuple: (mapped_data, hodl_profit, open_fee, close_fee)
    """
    fee_discount = 0.25 if token in ['SOL', 'ETH', 'BTC'] else 1
    initial_price = mapped_data['close'].iloc[0]
    final_price = mapped_data['close'].iloc[-1]
    initial_token_amount = entry_amount / initial_price

    # Calculate open fee
    open_fee = entry_amount * leverage * 0.001 * fee_discount
    
    # Calculate close fee based on final position value
    final_position_value = entry_amount * leverage * (final_price / initial_price)
    close_fee = final_position_value * 0.001 * fee_discount

    # Calculate hourly fees based on current token price
    mapped_data['hourly_fee'] = mapped_data['fundingRate'] * initial_token_amount * mapped_data['close'] * leverage
    if margin_direction == "Short":
        mapped_data['hourly_fee'] = -mapped_data['hourly_fee']

    mapped_data['cumulative_hourly_fee'] = mapped_data['hourly_fee'].cumsum()
    
    # Include close fee in total fee calculation at the last row
    mapped_data['total_fee'] = mapped_data['cumulative_hourly_fee'] + open_fee
    mapped_data.loc[mapped_data.index[-1], 'total_fee'] += close_fee

    mapped_data['price_change_pct'] = mapped_data['close'].pct_change()
    mapped_data['hourly_pnl'] = entry_amount * leverage * mapped_data['price_change_pct']
    if margin_direction == "Short":
        mapped_data['hourly_pnl'] = -mapped_data['hourly_pnl']
    
    mapped_data['cumulative_pnl'] = mapped_data['hourly_pnl'].cumsum()
    mapped_data['account_balance'] = entry_amount + mapped_data['cumulative_pnl'] - mapped_data['total_fee']

    # Calculate HODL profit
    hodl_profit = entry_amount * (final_price - initial_price) / initial_price

    return mapped_data, hodl_profit, open_fee, close_fee

# Create fee chart
def create_fee_chart(results):
    fig = go.Figure()
    for result in results:
        fig.add_trace(go.Scatter(x=result['data']['timestamp'], y=result['data']['total_fee'],
                                 mode='lines', name=f'{result["leverage"]}x Leverage'))
    fig.update_layout(title='Total Fees Over Time',
                      xaxis_title='Time', yaxis_title='Total Fees (USD)')
    return fig

# Create hourly fee chart
def create_hourly_fee_chart(results):
    fig = go.Figure()
    for result in results:
        fig.add_trace(go.Scatter(x=result['data']['timestamp'], y=result['data']['hourly_fee'],
                                 mode='lines', name=f'{result["leverage"]}x Leverage'))
    fig.update_layout(title='Hourly Fees Over Time',
                      xaxis_title='Time', yaxis_title='Hourly Fees (USD)')
    return fig

# Create account balance chart
def create_account_balance_chart(results):
    fig = go.Figure()
    for result in results:
        fig.add_trace(go.Scatter(x=result['data']['timestamp'], y=result['data']['account_balance'],
                                 mode='lines', name=f'{result["leverage"]}x Leverage'))
    fig.update_layout(title='Account Balance Over Time',
                      xaxis_title='Time', yaxis_title='Account Balance (USD)')
    return fig

# Calculate Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252  # Assuming 252 trading days in a year
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

# Add Sharpe Ratio to results
def add_sharpe_ratio(result):
    data = result['data']
    daily_returns = data['account_balance'].pct_change().dropna()
    sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    return sharpe_ratio

# Validate data
def validate_data(mapped_data, start_date, end_date):
    if mapped_data.empty:
        st.error("No data available for the selected date range.")
        return False
    
    data_start = mapped_data['timestamp'].min().date()
    data_end = mapped_data['timestamp'].max().date()
    
    if data_start > start_date or data_end < end_date:
        st.warning(f"Data only available from {data_start} to {data_end}. Adjusting date range.")
        return False
    
    return True

# Main app logic
def main():
    # Sidebar inputs
    token = st.sidebar.selectbox("Select Token", ["SOL", "BTC", "ETH"])
    margin_direction = st.sidebar.radio("Margin Direction", ["Long", "Short"])
    
    # Calculate default date range
    end_date = date.today()
    start_date = end_date - timedelta(days=14)  # 15 days ago (14 days difference)
    
    # Date inputs with new defaults
    start_date = st.sidebar.date_input("Start Date", value=start_date)
    end_date = st.sidebar.date_input("End Date", value=end_date)
    entry_amount = st.sidebar.number_input("Entry Amount (USD)", min_value=100, value=1000, step=100)

    # Leverage options
    leverage_options = [1.5, 2, 3, 4, 5, 6, 8, 10]
    use_custom_leverage = st.sidebar.checkbox("Use custom leverage")
    if use_custom_leverage:
        custom_leverage = st.sidebar.number_input("Enter custom leverage", min_value=1.0, max_value=20.0, value=1.5, step=0.1)
        selected_leverages = [custom_leverage]
    else:
        selected_leverages = []
        st.sidebar.write("Select Leverage Options:")
        for lev in leverage_options:
            if st.sidebar.checkbox(f"{lev}x", value=True):
                selected_leverages.append(lev)

    # Load and preprocess data
    funding_rates = load_funding_rates(token)
    funding_rates = preprocess_funding_rates(funding_rates)
    price_data = fetch_price_data(token, start_date, end_date)
    
    # Map funding rates to price data
    mapped_data = map_funding_rates_to_price(funding_rates, price_data)

    # Validate data
    if not validate_data(mapped_data, start_date, end_date):
        st.stop()

    # Calculate fees and profits for each leverage
    results = []
    for leverage in selected_leverages:
        data, hodl_profit, open_fee, close_fee = calculate_fees_and_profits(mapped_data.copy(), entry_amount, leverage, margin_direction, token)
        results.append({
            'leverage': leverage,
            'data': data,
            'hodl_profit': hodl_profit,
            'open_fee': open_fee,
            'close_fee': close_fee
        })

    # Add Sharpe Ratio to results
    for result in results:
        result['sharpe_ratio'] = add_sharpe_ratio(result)

    # Display charts
    if results:
        st.subheader("Fee Analysis")
        fee_chart = create_fee_chart(results)
        st.plotly_chart(fee_chart, use_container_width=True)

        st.subheader("Hourly Fee Analysis")
        hourly_fee_chart = create_hourly_fee_chart(results)
        st.plotly_chart(hourly_fee_chart, use_container_width=True)

        st.subheader("Account Balance Analysis")
        use_log_scale = st.checkbox("Use logarithmic scale for Account Balance")
        balance_chart = create_account_balance_chart(results)
        if use_log_scale:
            balance_chart.update_layout(yaxis_type="log")
        st.plotly_chart(balance_chart, use_container_width=True)

        # Funding Rates Chart
        st.subheader("Funding Rates")
        fig_funding = px.line(mapped_data, x='timestamp', y='fundingRate', title=f"Funding Rates for {token}")
        st.plotly_chart(fig_funding, use_container_width=True)

        # Fee Simulation and Profit Comparison
        st.subheader("Fee Simulation and Profit Comparison")

        comparison_data = []
        for result in results:
            leverage = result['leverage']
            data = result['data']
            total_fee = data['total_fee'].iloc[-1]
            final_balance = data['account_balance'].iloc[-1]
            profit = final_balance - entry_amount

            comparison_data.append({
                'Leverage': f'{leverage}x',
                'Total Fee': total_fee,
                'Final Balance': final_balance,
                'Profit': profit,
                'Sharpe Ratio': result['sharpe_ratio']
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df.set_index('Leverage'))

        # Profit Comparison Chart
        fig_profit_comparison = px.bar(comparison_df, x='Leverage', y='Profit', 
                                       title=f"Profit Comparison for Different Leverage ({token})")
        st.plotly_chart(fig_profit_comparison, use_container_width=True)

        # HODL vs All Leverage Strategies Profit Comparison
        st.subheader("HODL vs All Leverage Strategies Profit Comparison")
        hodl_comparison = pd.DataFrame({
            'Strategy': ['HODL'] + [f'{lev}x' for lev in selected_leverages],
            'Profit': [results[0]['hodl_profit']] + comparison_df['Profit'].tolist()
        })

        fig_hodl_comparison = px.bar(hodl_comparison, x='Strategy', y='Profit', 
                                     title=f"HODL vs All Leverage Strategies Profit Comparison for {token}")
        st.plotly_chart(fig_hodl_comparison, use_container_width=True)

        # Sharpe Ratio Comparison
        st.subheader("Sharpe Ratio Comparison")
        fig_sharpe_comparison = px.bar(comparison_df, x='Leverage', y='Sharpe Ratio', 
                                       title=f"Sharpe Ratio Comparison for Different Leverage ({token})")
        st.plotly_chart(fig_sharpe_comparison, use_container_width=True)

    # Price Chart
    st.subheader("Price Chart")
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(x=mapped_data['timestamp'],
                                       open=mapped_data['open'],
                                       high=mapped_data['high'],
                                       low=mapped_data['low'],
                                       close=mapped_data['close'],
                                       name='Price'))
    fig_price.update_layout(title=f"{token} Price", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig_price, use_container_width=True)

    # Display debug table
    st.subheader("Debug Table")
    if results:
        debug_df = results[0]['data'][['timestamp', 'close', 'fundingRate', 'hourly_fee', 'cumulative_hourly_fee', 'total_fee', 'price_change_pct', 'hourly_pnl', 'cumulative_pnl', 'account_balance']]
        
        # Add open and close fees to the debug table
        debug_df['open_fee'] = results[0]['open_fee']
        debug_df['close_fee'] = results[0]['close_fee']
        
        # Reorder columns for better readability
        debug_df = debug_df[['timestamp', 'close', 'fundingRate', 'open_fee', 'hourly_fee', 'close_fee', 'cumulative_hourly_fee', 'total_fee', 'price_change_pct', 'hourly_pnl', 'cumulative_pnl', 'account_balance']]
        
        st.dataframe(debug_df)
    else:
        st.write("No data to display. Please select at least one leverage option.")

if __name__ == "__main__":
    main()