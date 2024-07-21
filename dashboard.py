import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta, date
import json

# Set page config
st.set_page_config(page_title="Drift Protocol Fee Analysis", layout="wide")

# Title
st.title("Drift Protocol Fee and Profit Analysis")

# Sidebar
st.sidebar.header("Settings")

# Load funding rate data
@st.cache_data
def load_funding_rates(token):
    df = pd.read_csv(f"{token.lower()}.csv", sep=";")
    df['ts'] = pd.to_datetime(df['ts'], format="%m/%d/%Y, %H:%M:%S")
    return df

# Fetch price data
@st.cache_data
def fetch_price_data(token, start_date, end_date):
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
    data = json.loads(response.text)
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(data['t'], unit='s'),
        'open': data['o'],
        'high': data['h'],
        'low': data['l'],
        'close': data['c']
    })
    return df

# Calculate fees and profits
def calculate_fees_and_profits(funding_rates, price_data, position_size, leverage, margin_direction, token):
    fee_discount = 0.25 if token in ['SOL', 'ETH', 'BTC'] else 1
    open_fee = position_size * leverage * 0.001 * fee_discount  # 0.1% open fee with potential discount
    close_fee = position_size * leverage * 0.001 * fee_discount  # 0.1% close fee with potential discount
    
    if margin_direction == "Long":
        total_variable_fee = funding_rates['fundingRate'].sum() * position_size * leverage
    else:  # Short
        total_variable_fee = -funding_rates['fundingRate'].sum() * position_size * leverage
    
    total_fee = open_fee + close_fee + total_variable_fee
    
    initial_price = price_data['close'].iloc[0]
    final_price = price_data['close'].iloc[-1]
    price_change = (final_price - initial_price) / initial_price
    
    hodl_profit = position_size * price_change
    leverage_profit = (position_size * leverage * price_change) - total_fee
    
    if margin_direction == "Short":
        leverage_profit = -leverage_profit
    
    return open_fee, close_fee, total_variable_fee, total_fee, hodl_profit, leverage_profit

# Main app logic
def main():
    # Sidebar inputs
    token = st.sidebar.selectbox("Select Token", ["SOL", "BTC", "ETH"])
    margin_direction = st.sidebar.radio("Margin Direction", ["Long", "Short"])
    start_date = st.sidebar.date_input("Start Date", date(2024, 6, 18))
    end_date = st.sidebar.date_input("End Date", date(2024, 7, 18))
    position_size = st.sidebar.number_input("Position Size (USD)", min_value=100, value=1000, step=100)

    # Leverage options
    leverage_options = [1.5, 2, 3, 4, 5, 6, 8, 10]
    selected_leverages = []
    st.sidebar.write("Select Leverage Options:")
    for lev in leverage_options:
        if st.sidebar.checkbox(f"{lev}x", value=True):
            selected_leverages.append(lev)

    # Load data
    funding_rates = load_funding_rates(token)
    price_data = fetch_price_data(token, start_date, end_date)
    
    # Filter data based on date range
    mask = (funding_rates['ts'] >= pd.Timestamp(start_date)) & (funding_rates['ts'] <= pd.Timestamp(end_date))
    funding_rates_filtered = funding_rates.loc[mask]

    # Funding Rates Chart
    st.header("Funding Rates")
    fig_funding = px.line(funding_rates_filtered, x='ts', y='fundingRate', title=f"Funding Rates for {token}")
    st.plotly_chart(fig_funding, use_container_width=True)

    # Fee Simulation and Profit Comparison
    st.header("Fee Simulation and Profit Comparison")

    comparison_data = []

    for leverage in selected_leverages:
        open_fee, close_fee, total_variable_fee, total_fee, hodl_profit, leverage_profit = calculate_fees_and_profits(
            funding_rates_filtered, price_data, position_size, leverage, margin_direction, token
        )
        
        comparison_data.append({
            'Leverage': f'{leverage}x',
            'Open Fee': open_fee,
            'Close Fee': close_fee,
            'Total Variable Fee': total_variable_fee,
            'Total Fee': total_fee,
            'Profit': leverage_profit
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df.set_index('Leverage'))

    # Profit Comparison Chart
    fig_profit_comparison = px.bar(comparison_df, x='Leverage', y='Profit', 
                                   title=f"Profit Comparison for Different Leverage ({token})")
    st.plotly_chart(fig_profit_comparison, use_container_width=True)

    # Fee Breakdown Chart
    fee_breakdown = comparison_df.melt(id_vars=['Leverage'], 
                                       value_vars=['Open Fee', 'Close Fee', 'Total Variable Fee'],
                                       var_name='Fee Type', value_name='Amount')
    fig_fee_breakdown = px.bar(fee_breakdown, x='Leverage', y='Amount', color='Fee Type', 
                               title=f"Fee Breakdown for Different Leverage ({token})",
                               barmode='stack')
    st.plotly_chart(fig_fee_breakdown, use_container_width=True)

    # HODL vs All Leverage Strategies Profit Comparison
    st.header("HODL vs All Leverage Strategies Profit Comparison")
    hodl_comparison = pd.DataFrame({
        'Strategy': ['HODL'] + [f'{lev}x' for lev in selected_leverages],
        'Profit': [hodl_profit] + comparison_df['Profit'].tolist()
    })

    fig_hodl_comparison = px.bar(hodl_comparison, x='Strategy', y='Profit', 
                                 title=f"HODL vs All Leverage Strategies Profit Comparison for {token}")
    st.plotly_chart(fig_hodl_comparison, use_container_width=True)

    # Price Chart
    st.header("Price Chart")
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(x=price_data['timestamp'],
                                       open=price_data['open'],
                                       high=price_data['high'],
                                       low=price_data['low'],
                                       close=price_data['close'],
                                       name='Price'))
    fig_price.update_layout(title=f"{token} Price", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig_price, use_container_width=True)

if __name__ == "__main__":
    main()