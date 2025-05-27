# AI Day Trading Bot

This project is an AI-powered day trading bot using Alpaca's paper trading API. It features:

- Real-time stock data ingestion
- Machine learning signal generation (SVM)
- Live paper trading execution
- Backtesting capabilities
- Streamlit dashboard for live monitoring
- Telegram notifications

## Setup

1. Rename `.env.example` to `.env` and add your API keys.
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run the bot:
   ```
   python trading_bot.py
   ```
4. Run the dashboard:
   ```
   streamlit run dashboard.py
   ```
