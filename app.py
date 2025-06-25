import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests
from datetime import datetime, timedelta
import os

st.write("🗂 Current working directory:", os.getcwd())
st.write("📁 Files in current directory:", os.listdir())

st.set_page_config(layout="wide")
st.title("📈 Hybrid Stock Price Predictor (LSTM + Technical + Fundamental Analysis)")

API_KEY = "BISBYX9HTP79QMXV"

# ---------------- Sidebar ----------------
st.sidebar.header("Enter Stock Symbol")
symbol = st.sidebar.text_input("Example: GOOG, AAPL, TSLA", "GOOG").strip().upper()

# Year-wise dropdown selection
period_options = {
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "10 Years": "10y",
    "15 Years": "15y"
}
selected_period_label = st.sidebar.selectbox("Select data period", list(period_options.keys()), index=4)
selected_period = period_options[selected_period_label]

# ---------------- Fetch Stock Data ----------------
@st.cache_data(ttl=86400)
def fetch_stock_data(symbol, period):
    try:
        df = yf.download(symbol, period=period)
        if df.empty:
            st.error(f"❌ yfinance returned empty data for symbol: {symbol}")
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if 'Close' in df.columns and not df[['Close']].empty:
            st.success("✅ Data fetched from yfinance and cached (in memory).")
            return df[['Close']]
        else:
            st.error("❌ 'Close' column missing in yfinance data.")
            st.write("Available columns:", list(df.columns))
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Exception during yfinance fetch: {e}")
        return pd.DataFrame()


# ---------------- Technical Indicators ----------------
def compute_indicators(data):
    data['Rolling_Mean'] = data['Close'].rolling(window=30).mean()
    data['Rolling_STD'] = data['Close'].rolling(window=30).std()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']

    delta = data['Close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = loss.abs()
    roll_up = gain.rolling(window=14).mean()
    roll_down = loss.rolling(window=14).mean()
    rs = roll_up / roll_down
    data['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    data['MA_100'] = data['Close'].rolling(window=100).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()

    return data.dropna()


# ---------------- Fundamentals ----------------
def get_fundamentals(symbol):
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={API_KEY}'
    try:
        overview = requests.get(url).json()
        pe = float(overview.get('PERatio', 0))
        pb = float(overview.get('PriceToBookRatio', 0))
        roe = float(overview.get('ReturnOnEquityTTM', 0))
        mc = float(overview.get('MarketCapitalization', 0)) / 1e9
        evals = [
            "Good" if pe <= 30 else "Bad",
            "Good" if pb <= 10 else "Bad",
            "Good" if roe >= 15 else "Bad",
            "Good" if mc >= 10 else "Bad"
        ]
        return [pe, pb, roe, mc], evals
    except:
        return [0, 0, 0, 0], ["Unavailable"] * 4


# ---------------- LSTM Preparation ----------------
def prepare_lstm_data(data, scaler, steps=100):
    scaled_data = scaler.fit_transform(data[['Close']])
    x, y = [], []
    for i in range(steps, len(scaled_data)):
        x.append(scaled_data[i - steps:i])
        y.append(scaled_data[i, 0])
    return np.array(x), np.array(y)


# ---------------- Main Logic ----------------
if symbol:
    st.info(f"🔄 Fetching and analyzing data for: **{symbol}** ({selected_period_label})")
    try:
        raw_data = fetch_stock_data(symbol, selected_period)
        if raw_data.empty:
            st.error("❌ No data found for the provided stock symbol.")
            st.stop()

        data = compute_indicators(raw_data.copy())

        # --------- Charts: Price + MACD + RSI ----------
        st.subheader("📉 Stock Price, MACD & RSI (TradingView-style)")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

        # -- Price Chart --
        ax1.plot(data.index, data['Close'], color='cyan', label='Close Price')
        ax1.set_title('Close Price', fontsize=14, fontweight='bold')
        ax1.set_ylabel("Price")
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(loc="upper left")

        # -- MACD --
        macd_line = data['MACD']
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal_line
        ax2.bar(data.index, hist, color=['green' if val >= 0 else 'red' for val in hist], alpha=0.4, width=1)
        ax2.plot(data.index, macd_line, label='MACD', color='blue')
        ax2.plot(data.index, signal_line, label='Signal', color='orange')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_title("MACD (12, 26, 9)", fontsize=14, fontweight='bold')
        ax2.set_ylabel("MACD")
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(loc="upper left")

        # -- RSI --
        ax3.plot(data.index, data['RSI'], label='RSI (14)', color='violet')
        ax3.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        ax3.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        ax3.set_ylim(0, 100)
        ax3.set_title("Relative Strength Index (RSI)", fontsize=14, fontweight='bold')
        ax3.set_ylabel("RSI")
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.legend(loc="upper left")

        plt.tight_layout()
        st.pyplot(fig)

        # --------- Moving Averages ----------
        st.subheader("📊 100-Day & 200-Day Moving Averages")
        fig2, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data.index, data['Close'], label='Close Price')
        ax.plot(data.index, data['MA_100'], label='100-Day MA', color='orange')
        ax.plot(data.index, data['MA_200'], label='200-Day MA', color='green')
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig2)

        # --------- LSTM Prediction ----------
        train_size = int(len(data) * 0.8)
        train, test = data[:train_size], data[train_size:]
        scaler = MinMaxScaler()
        x_train, y_train = prepare_lstm_data(train, scaler)
        x_test, y_test = prepare_lstm_data(pd.concat([train[-100:], test]), scaler)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        try:
            model_path = os.path.join(os.getcwd(), "lstm_stock_predictor.h5")
            model = load_model(model_path")
            st.success("✅ Pre-trained model loaded successfully.")

            y_pred_scaled = model.predict(x_test)
            y_pred = scaler.inverse_transform(y_pred_scaled)
            y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

            st.subheader("📈 Actual vs Predicted Prices with Moving Averages")
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            ax3.plot(y_actual, label="Actual", color='blue')
            ax3.plot(y_pred, label="Predicted", color='red')
            ax3.plot(data['MA_100'].iloc[-len(y_pred):].values, label="100-Day MA", color='orange')
            ax3.plot(data['MA_200'].iloc[-len(y_pred):].values, label="200-Day MA", color='green')
            ax3.set_ylabel("Price")
            ax3.legend()
            st.pyplot(fig3)

            # Metrics
            mse = mean_squared_error(y_actual, y_pred)
            mae = mean_absolute_error(y_actual, y_pred)
            r2 = r2_score(y_actual, y_pred)
            mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
            tech_accuracy = 100 - mape

            st.write(f"**MSE:** {mse:.4f} | **MAE:** {mae:.4f} | **R² Score:** {r2:.4f} | **Technical Accuracy:** {tech_accuracy:.2f}%")

        except Exception as e:
            st.error("❌ Error loading `.h5` model. Make sure `lstm_stock_predictor.h5` is present in your project folder.")
            st.stop()

        # --------- Fundamental Analysis ----------
        st.subheader("📚 Fundamental Analysis")
        fundamentals, fund_eval = get_fundamentals(symbol)
        fund_df = pd.DataFrame({
            "Metric": ["PE Ratio", "PB Ratio", "ROE", "Market Cap (B)"],
            "Value": fundamentals,
            "Evaluation": fund_eval
        })
        st.table(fund_df)

        # --------- Recommendation ----------
        fund_score = sum([1 if val == "Good" else 0 for val in fund_eval])
        fund_strong = fund_score >= 3
        tech_strong = r2 >= 0.75 and tech_accuracy >= 80

        st.subheader("📌 Recommendation")
        if fund_strong:
            st.success("✅ Fundamentals: Strong (✓ at least 3 out of 4 metrics passed)")
        else:
            st.warning("⚠️ Fundamentals: Weak")

        if tech_strong:
            st.success("✅ Technical Prediction: Strong")
        else:
            st.warning("⚠️ Technical Prediction: Weak")

        if fund_strong and tech_strong:
            st.success("📈 Final Verdict: Suitable for investment")
        elif fund_strong:
            st.info("📉 Final Verdict: Monitor technicals before investing")
        elif tech_strong:
            st.info("📉 Final Verdict: Investigate company fundamentals further")
        else:
            st.error("❌ Final Verdict: Not recommended currently")

        combined_accuracy = ((fund_score / 4) * 100 + tech_accuracy) / 2
        st.info(f"🔢 Combined Accuracy: **{combined_accuracy:.2f}%**")

    except Exception as e:
        st.error(f"An error occurred while processing the data:\n\n**{e}**")
