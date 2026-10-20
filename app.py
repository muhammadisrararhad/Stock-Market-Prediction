import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib

# Page configuration
st.set_page_config(
    page_title="Stock Trend Predictor",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Stock Price Direction Predictor")
st.write("This app uses a trained **Random Forest Classifier** to predict whether a stock's closing price will go up or down on the next trading day.")

# Load the trained model and features
@st.cache_resource
def load_model():
    model = joblib.load('stock_random_forest_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

try:
    model, features = load_model()
except FileNotFoundError:
    st.error("Model files not found! Please make sure 'stock_random_forest_model.pkl' and 'model_features.pkl' are in the same directory.")
    st.stop()

# Sidebar inputs for user selection
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()

# Fetch latest data for the ticker to show current stats
@st.cache_data
def get_latest_data(stock_ticker):
    data = yf.download(stock_ticker, period="1mo")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

try:
    df_live = get_latest_data(ticker)
    latest_close = df_live['Close'].iloc[-1]
    st.sidebar.info(f"Latest {ticker} Close Price: ${latest_close:.2f}")
except Exception as e:
    st.sidebar.error(f"Could not fetch data for {ticker}. Check ticker symbol.")
    st.stop()

# Compute latest features from the fetched data for prediction
df_live['Daily_Return'] = df_live['Close'].pct_change()
df_live['Lag_Return'] = df_live['Daily_Return'].shift(1)
df_live['Rolling_Vol_5'] = df_live['Daily_Return'].rolling(window=5).std()
df_live['Rolling_Mean_5'] = df_live['Daily_Return'].rolling(window=5).mean()
df_live['High_Low_Spread'] = (df_live['High'] - df_live['Low']) / df_live['Low']
df_live['Close_Open_Spread'] = (df_live['Close'] - df_live['Open']) / df_live['Open']

# Extract the most recent row of features
latest_features = df_live[features].iloc[-1].values.reshape(1, -1)

# Main Dashboard Interface
if st.button("Predict Next Day Movement"):
    prediction = model.predict(latest_features)[0]
    probabilities = model.predict_proba(latest_features)[0]
    
    st.write("---")
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.success(f"**Prediction:** Bullish 🟢 (Tomorrow's Close is likely to be HIGHER than today's)")
    else:
        st.error(f"**Prediction:** Bearish 🔴 (Tomorrow's Close is likely to be LOWER or equal)")
        
    st.write(f"**Confidence:** Model predicts with **{probabilities[prediction]*100:.2f}%** probability.")
    
    # Display recent chart
    st.write("---")
    st.subheader(f"Recent Closing Prices for {ticker}")
    st.line_chart(df_live['Close'].tail(30))
