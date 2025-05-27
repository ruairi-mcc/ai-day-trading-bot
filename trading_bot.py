import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame
from sklearn.svm import SVC
from telegram_config import send_telegram_message

load_dotenv()

api = REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    base_url='https://paper-api.alpaca.markets'
)

symbol = 'AAPL'
df = api.get_bars(symbol, TimeFrame.Minute, limit=1000).df
df['return'] = df['close'].pct_change()

for i in range(1, 6):
    df[f'lag_{i}'] = df['return'].shift(i)
df['direction'] = np.where(df['return'] > 0, 1, 0)
df.dropna(inplace=True)

X = df[[f'lag_{i}' for i in range(1, 6)]]
y = df['direction']
model = SVC()
model.fit(X, y)

while True:
    try:
        latest = api.get_bars(symbol, TimeFrame.Minute, limit=6).df
        latest['return'] = latest['close'].pct_change()
        features = latest['return'].dropna().values[-5:][::-1].reshape(1, -1)

        if features.shape[1] != 5:
            continue

        signal = model.predict(features)[0]
        try:
            position = int(api.get_position(symbol).qty)
        except:
            position = 0

        if signal == 1 and position == 0:
            api.submit_order(symbol=symbol, qty=1, side='buy', type='market', time_in_force='gtc')
            send_telegram_message("🟢 BUY signal executed.")
        elif signal == 0 and position > 0:
            api.submit_order(symbol=symbol, qty=1, side='sell', type='market', time_in_force='gtc')
            send_telegram_message("🔴 SELL signal executed.")
        else:
            print("HOLD")

        time.sleep(60)
    except Exception as e:
        send_telegram_message(f"⚠️ Error: {e}")
        time.sleep(60)
