import os
import time
import logging
import yaml
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Alpaca imports
from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import OrderRequest
from alpaca.trading.enums import OrderSide, OrderType, AssetClass

# Telegram
try:
    from telegram import Bot
except ImportError:
    Bot = None

# ─── CONFIGURATION ──────────────────────────────────────────────
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
load_dotenv()

API_KEY    = os.getenv("ALPACA_API_KEY", config["alpaca"]["api_key"])
API_SECRET = os.getenv("ALPACA_API_SECRET", config["alpaca"]["api_secret"])
PAPER_MODE = config["alpaca"]["paper"]

REQUEST_SYMBOL = config["trading"]["symbol"]
SHORT_SMA      = config["trading"]["short_sma"]
LONG_SMA       = config["trading"]["long_sma"]
TRADE_QTY      = str(config["trading"]["trade_qty"])
FEE_RATE       = config["trading"]["fee_rate"]
INITIAL_CASH   = config["trading"]["initial_cash"]
STOP_LOSS_PCT  = config["trading"]["stop_loss_pct"]
MAX_DRAWDOWN_PCT = config["trading"]["max_drawdown_pct"]
USE_LIMIT_ORDERS = config["trading"]["use_limit_orders"]
DRY_RUN        = config["trading"]["dry_run"]

TELEGRAM_ENABLED = config["telegram"]["enabled"]
TELEGRAM_TOKEN   = config["telegram"]["token"]
TELEGRAM_CHAT_ID = config["telegram"]["chat_id"]

BACKTEST_HOUR = 500

# ─── LOGGING SETUP ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ─── Telegram Notification ─────────────────────────────────────
def send_telegram(msg):
    if TELEGRAM_ENABLED and Bot is not None:
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        except Exception as e:
            logging.error(f"Telegram notification failed: {e}")

# ─── Alpaca Clients ─────────────────────────────────────────────
def init_alpaca_clients():
    try:
        hist_client = CryptoHistoricalDataClient(API_KEY, API_SECRET)
        trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER_MODE)
        logging.info("Alpaca clients initialized successfully.")
        return hist_client, trading_client
    except Exception as e:
        logging.error(f"Failed to initialize Alpaca clients: {e}")
        exit(1)

hist_client, trading_client = init_alpaca_clients()

# ─── Helper: Fetch Crypto Bars ─────────────────────────────────
def fetch_crypto_bars(request_symbol: str, start, end, timeframe: TimeFrame) -> pd.DataFrame:
    try:
        if isinstance(start, datetime):
            start = start.isoformat()
        if isinstance(end, datetime):
            end = end.isoformat()

        req = CryptoBarsRequest(
            symbol_or_symbols=[request_symbol],
            timeframe=timeframe,
            start=start,
            end=end,
        )
        raw = hist_client.get_crypto_bars(req)
        df_all = raw.df

        if df_all.empty:
            logging.warning(f"No data returned for {request_symbol} between {start} and {end}.")
            return pd.DataFrame()

        # If columns are a MultiIndex, extract the symbol
        if isinstance(df_all.columns, pd.MultiIndex):
            symbol_returned = df_all.columns.levels[0][0]
            data = df_all[symbol_returned].copy()
        else:
            data = df_all.copy()

        data = data.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index(level=1, drop=True)
        data.index.name = "ts"
        logging.info(f"Fetched {len(data)} bars for {symbol_returned if 'symbol_returned' in locals() else request_symbol}.")
        return data
    except Exception as e:
        logging.error(f"Error fetching crypto bars: {e}")
        return pd.DataFrame()

# ─── Risk Management ───────────────────────────────────────────
def check_stop_loss(entry_price, current_price):
    if entry_price is None:
        return False
    return current_price < entry_price * (1 - STOP_LOSS_PCT)

def check_max_drawdown(equity_curve):
    if not equity_curve:
        return False
    equity = [e['equity'] for e in equity_curve]
    peak = max(equity)
    trough = min(equity)
    drawdown = (peak - trough) / peak
    return drawdown > MAX_DRAWDOWN_PCT

# ─── Position Sizing ───────────────────────────────────────────
def calculate_position_size(cash, price, risk_pct=0.01):
    # Risk 1% of cash per trade by default
    risk_amount = cash * risk_pct
    qty = risk_amount / price
    return round(qty, 6)

# ─── Backtest: SMA Crossover ──────────────────────────────────
def backtest_sma_crossover(hours: int = BACKTEST_HOUR):
    now_utc = datetime.utcnow()
    end_dt = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    start_dt = (now_utc - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

    df = fetch_crypto_bars(REQUEST_SYMBOL, start=start_dt, end=end_dt, timeframe=TimeFrame.Hour)
    if df.empty or len(df) < LONG_SMA + 2:
        logging.warning("Not enough data to run backtest.")
        return None, None

    df["SMA_short"] = df["Close"].rolling(window=SHORT_SMA).mean()
    df["SMA_long"]  = df["Close"].rolling(window=LONG_SMA).mean()
    df = df.dropna().copy()

    df["signal"] = np.where(df["SMA_short"] > df["SMA_long"], 1, -1)
    df["signal"] = df["signal"].shift(1)
    df = df.dropna().copy()

    cash = INITIAL_CASH
    btc_pos = 0.0
    equity_curve = []
    trade_log = []
    total_fees = 0.0
    entry_price = None
    max_equity = INITIAL_CASH

    for ts, row in df.iterrows():
        price = float(row["Close"])
        sig   = int(row["signal"])

        # Stop-loss check
        if btc_pos > 0 and check_stop_loss(entry_price, price):
            proceeds = price * btc_pos * (1 - FEE_RATE)
            fee = price * btc_pos * FEE_RATE
            cash += proceeds
            total_fees += fee
            trade_log.append({"ts": ts, "action": "STOP-LOSS", "price": price, "qty": btc_pos, "cash": cash, "equity": cash})
            btc_pos = 0.0
            entry_price = None

        # BUY
        elif sig == 1 and btc_pos == 0:
            cost = price * float(TRADE_QTY) * (1 + FEE_RATE)
            if cash >= cost:
                cash -= cost
                btc_pos += float(TRADE_QTY)
                fee = price * float(TRADE_QTY) * FEE_RATE
                total_fees += fee
                entry_price = price
                trade_log.append({"ts": ts, "action": "BUY", "price": price, "qty": float(TRADE_QTY), "cash": cash, "equity": cash + btc_pos * price})

        # SELL
        elif sig == -1 and btc_pos > 0:
            proceeds = price * btc_pos * (1 - FEE_RATE)
            fee = price * btc_pos * FEE_RATE
            cash += proceeds
            total_fees += fee
            trade_log.append({"ts": ts, "action": "SELL", "price": price, "qty": btc_pos, "cash": cash, "equity": cash})
            btc_pos = 0.0
            entry_price = None

        equity = cash + btc_pos * price
        equity_curve.append({"ts": ts, "equity": equity})
        max_equity = max(max_equity, equity)

        # Max drawdown check
        if check_max_drawdown(equity_curve):
            logging.warning("Max drawdown exceeded. Stopping backtest.")
            break

    history_df = pd.DataFrame(equity_curve)
    if isinstance(history_df['ts'].iloc[0], tuple):
        history_df['ts'] = [t[0] for t in history_df['ts']]
    if not pd.api.types.is_datetime64_any_dtype(history_df['ts']):
        history_df['ts'] = pd.to_datetime(history_df['ts'], errors='coerce')
    history_df = history_df.set_index("ts")

    trades_df = pd.DataFrame(trade_log)
    logging.info(f"Backtest complete. Final equity: ${history_df['equity'].iloc[-1]:.2f}, Total fees: ${total_fees:.2f}")
    return df, history_df

# ─── Plot Backtest ────────────────────────────────────────────
import matplotlib.dates as mdates

def plot_backtest(history_df: pd.DataFrame, save_path: str = None):
    if history_df is None or history_df.empty:
        logging.warning("No equity history to plot.")
        return

    final_equity = history_df["equity"].iloc[-1]
    logging.info(f"Final equity (backtest): ${final_equity:.2f}")

    plt.figure(figsize=(10, 4))
    plt.plot(history_df.index, history_df["equity"], label="Equity Curve", color="#00bcd4", linewidth=2)
    plt.axhline(INITIAL_CASH, color="gray", linestyle="--", label=f"Initial Capital (${INITIAL_CASH:,.0f})")
    plt.title(f"Backtest: {REQUEST_SYMBOL} {SHORT_SMA}h SMA vs {LONG_SMA}h SMA")
    plt.ylabel("Portfolio Value (USD)")
    plt.xlabel("Time (UTC)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if pd.api.types.is_datetime64_any_dtype(history_df.index):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))

    if save_path:
        plt.savefig(save_path, dpi=150)
        logging.info(f"Plot saved to {save_path}")
    else:
        plt.show()

# ─── Live Trading Loop ────────────────────────────────────────
def run_live_sma_loop():
    last_signal = 0  # 1 = long, -1 = flat
    entry_price = None
    max_equity = INITIAL_CASH
    cash = INITIAL_CASH
    btc_pos = 0.0

    logging.info("🚀 Entering live (paper) trading loop. Press Ctrl+C to exit.")

    try:
        while True:
            window_hours = max(LONG_SMA + SHORT_SMA, 300)
            now_utc = datetime.utcnow()
            end_dt = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            start_dt = (now_utc - timedelta(hours=window_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

            try:
                df = fetch_crypto_bars(REQUEST_SYMBOL, start=start_dt, end=end_dt, timeframe=TimeFrame.Hour)
                if df.empty or len(df) < LONG_SMA + 2:
                    logging.warning("Not enough data to compute SMAs. Skipping this cycle.")
                    time.sleep(60)
                    continue
            except Exception as e:
                logging.error(f"Data fetch failed: {e}")
                time.sleep(60)
                continue

            df["SMA_short"] = df["Close"].rolling(window=SHORT_SMA).mean()
            df["SMA_long"]  = df["Close"].rolling(window=LONG_SMA).mean()
            df = df.dropna().copy()

            if df.empty:
                logging.warning("No data after SMA calculation. Skipping this cycle.")
                time.sleep(60)
                continue

            latest = df.iloc[-1]
            signal = 1 if float(latest["SMA_short"]) > float(latest["SMA_long"]) else -1
            price  = float(latest["Close"])
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

            # Check current position (robustness)
            try:
                positions = trading_client.get_all_positions()
                btc_position = next((p for p in positions if p.symbol.replace("USDC", "USD") == REQUEST_SYMBOL.replace("/", "")), None)
                currently_long = btc_position is not None and float(btc_position.qty) > 0
            except Exception as e:
                logging.error(f"Could not fetch positions: {e}")
                currently_long = (last_signal == 1)

            # Stop-loss check (live)
            if currently_long and entry_price and price < entry_price * (1 - STOP_LOSS_PCT):
                if not DRY_RUN:
                    order = OrderRequest(
                        symbol=REQUEST_SYMBOL,
                        qty=TRADE_QTY,
                        side=OrderSide.SELL,
                        type=OrderType.MARKET,
                        time_in_force="gtc",
                        asset_class=AssetClass.CRYPTO,
                    )
                    try:
                        resp = trading_client.submit_order(order)
                        logging.info(f"{timestamp}  🛑 STOP-LOSS SELL {TRADE_QTY} BTC @ {price:.2f} USD → {resp}")
                        send_telegram(f"STOP-LOSS SELL {TRADE_QTY} BTC @ {price:.2f} USD")
                        last_signal = -1
                        entry_price = None
                    except Exception as e:
                        logging.error(f"{timestamp}  ❌ STOP-LOSS SELL failed: {e}")
                else:
                    logging.info(f"{timestamp}  [DRY RUN] 🛑 STOP-LOSS SELL {TRADE_QTY} BTC @ {price:.2f} USD")
                    send_telegram(f"[DRY RUN] STOP-LOSS SELL {TRADE_QTY} BTC @ {price:.2f} USD")
                    last_signal = -1
                    entry_price = None

            # BUY
            elif signal == 1 and not currently_long:
                if not DRY_RUN:
                    order = OrderRequest(
                        symbol=REQUEST_SYMBOL,
                        qty=TRADE_QTY,
                        side=OrderSide.BUY,
                        type=OrderType.MARKET,
                        time_in_force="gtc",
                        asset_class=AssetClass.CRYPTO,
                    )
                    try:
                        resp = trading_client.submit_order(order)
                        logging.info(f"{timestamp}  ▶️ BUY  {TRADE_QTY} BTC @ {price:.2f} USD → {resp}")
                        send_telegram(f"BUY {TRADE_QTY} BTC @ {price:.2f} USD")
                        last_signal = 1
                        entry_price = price
                    except Exception as e:
                        logging.error(f"{timestamp}  ❌ BUY failed: {e}")
                else:
                    logging.info(f"{timestamp}  [DRY RUN] ▶️ BUY  {TRADE_QTY} BTC @ {price:.2f} USD")
                    send_telegram(f"[DRY RUN] BUY {TRADE_QTY} BTC @ {price:.2f} USD")
                    last_signal = 1
                    entry_price = price

            # SELL
            elif signal == -1 and currently_long:
                if not DRY_RUN:
                    order = OrderRequest(
                        symbol=REQUEST_SYMBOL,
                        qty=TRADE_QTY,
                        side=OrderSide.SELL,
                        type=OrderType.MARKET,
                        time_in_force="gtc",
                        asset_class=AssetClass.CRYPTO,
                    )
                    try:
                        resp = trading_client.submit_order(order)
                        logging.info(f"{timestamp}  ▶️ SELL {TRADE_QTY} BTC @ {price:.2f} USD → {resp}")
                        send_telegram(f"SELL {TRADE_QTY} BTC @ {price:.2f} USD")
                        last_signal = -1
                        entry_price = None
                    except Exception as e:
                        logging.error(f"{timestamp}  ❌ SELL failed: {e}")
                else:
                    logging.info(f"{timestamp}  [DRY RUN] ▶️ SELL {TRADE_QTY} BTC @ {price:.2f} USD")
                    send_telegram(f"[DRY RUN] SELL {TRADE_QTY} BTC @ {price:.2f} USD")
                    last_signal = -1
                    entry_price = None
            else:
                logging.info(f"{timestamp}  — No trade. Current signal = {signal}")

            # Sleep until ~HH:02 UTC for next cycle
            now = datetime.utcnow()
            next_run = (now + timedelta(hours=1)).replace(minute=2, second=0, microsecond=0)
            secs_to_sleep = (next_run - now).total_seconds()
            mins_to_sleep = secs_to_sleep / 60
            logging.info(f"⏱️ Sleeping for {mins_to_sleep:.1f} minutes until next cycle…\n")
            time.sleep(max(secs_to_sleep, 60))  # Never sleep less than 60s

    except KeyboardInterrupt:
        logging.info("🛑 Bot terminated by user. Goodbye!")

# ─── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.info("=== BTC‐Only Alpaca SMA Crossover Bot ===")
    logging.info(f"Paper mode: {PAPER_MODE}")
    logging.info(f"Using API_KEY = {API_KEY[:4]}****, SECRET = {API_SECRET[:4]}****")

    # 1) Run backtest
    logging.info(f"1) Backtesting on the last {BACKTEST_HOUR} hours …")
    df_signals, hist_equity = backtest_sma_crossover()
    if hist_equity is not None and not hist_equity.empty:
        plot_backtest(hist_equity)
    else:
        logging.warning("Backtest failed or returned no data.")

    # 2) Start live (paper) trading loop
    logging.info("2) Starting live (paper) trading loop …\n")
    run_live_sma_loop()
