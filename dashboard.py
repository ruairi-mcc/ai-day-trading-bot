import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame
import os
from dotenv import load_dotenv

load_dotenv()

api = REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    base_url='https://paper-api.alpaca.markets'
)

st.title("📈 AI Trading Bot Dashboard")

symbol = "AAPL"
bars = api.get_bars(symbol, TimeFrame.Minute, limit=500).df
bars = bars[bars['symbol'] == symbol]

bars['return'] = bars['close'].pct_change()
bars['cumulative_return'] = (1 + bars['return']).cumprod()

st.subheader("Cumulative Return")
st.line_chart(bars['cumulative_return'])

st.subheader("Price Chart")
st.line_chart(bars['close'])

st.success("Dashboard updated with live data.")
