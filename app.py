# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf

# 1. Define tickers and download historical data (e.g., 2024 to 2026)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
print('Downloading stock data...')
stock_data = yf.download(tickers, start='2024-01-01', end='2026-01-01')[
    'Volume'
]

# --- 2. Exploratory Data Analysis (EDA) ---
print('\n--- 1. First 5 Rows of the Dataset ---')
print(stock_data.head())

print('\n--- 2. Dataset Information ---')
print(stock_data.info())

print('\n--- 3. Summary Statistics ---')
print(stock_data.describe())

print('\n--- 4. Checking for Missing Values ---')
print(stock_data.isnull().sum())

# --- 3. Data Preparation for Pie Chart ---
# Calculate the total trading volume for each stock over the given period
total_volume = stock_data.sum()

# --- 4. Visualizing with a Pie Chart ---
plt.figure(figsize=(8, 8))

# Define custom colors and explode the largest slice slightly for emphasis
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
explode = (0.05, 0, 0, 0)

plt.pie(
    total_volume,
    labels=total_volume.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    explode=explode,
    shadow=True,
)

plt.title(
    'Total Trading Volume Distribution (2024–2026)', fontsize=14, fontweight='bold'
)
plt.tight_layout()

# Display the chart
plt.show()


# 1. Install any required packages (just in case)
get_ipython().getoutput("pip install -q yfinance xgboost scikit-learn matplotlib pandas")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import yfinance as yf

print('Downloading multi-asset data (AAPL, S&P 500, VIX)...')

# 2. Download data
data = yf.download(['AAPL', '^GSPC', '^VIX'], start='2021-01-01', end='2026-01-01')[
    'Close'
]

# Handle multi-index columns if returned by yfinance
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.columns = ['AAPL', 'SP500', 'VIX']
df = data.dropna()

# --- 3. Feature Engineering ---
df['Return'] = df['AAPL'].pct_change()
df['SP_Return'] = df['SP500'].pct_change()
df['MA_5'] = df['AAPL'].rolling(window=5).mean()
df['MA_20'] = df['AAPL'].rolling(window=20).mean()

# Calculate RSI
delta = df['AAPL'].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
rs = gain.rolling(window=14).mean() / loss.rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + rs))

# Target: 1 if AAPL is higher 3 days from now
df['Target'] = (df['AAPL'].shift(-3) > df['AAPL']).astype(int)
df = df.dropna()

features = ['SP500', 'VIX', 'SP_Return', 'Return', 'MA_5', 'MA_20', 'RSI']
X = df[features]
y = df['Target']

# --- 4. Strict 80:20 Chronological Split ---
train_size = int(len(X) * 0.80)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f'Training Set Size: {len(X_train)} rows')
print(f'Testing Set Size: {len(X_test)} rows')

# --- 5. Train XGBoost Model ---
print('\nTraining XGBoost Model...')
model = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42
)
model.fit(X_train, y_train)

# --- 6. Evaluate ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'\nModel Accuracy: {accuracy * 100:.2f}%')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# --- 7. Plot Results inside Kaggle ---
plt.figure(figsize=(10, 4))
df['AAPL'].tail(100).plot(title='AAPL Recent Close Prices', color='blue')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()


# Install required libraries quietly
get_ipython().getoutput("pip install -q yfinance xgboost scikit-learn matplotlib pandas")

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import yfinance as yf

# Suppress future warnings from yfinance
warnings.simplefilter(action='ignore', category=FutureWarning)

print('Downloading multi-asset data (AAPL, S&P 500, VIX)...')

# Explicitly set auto_adjust to clear the warning
data = yf.download(
    ['AAPL', '^GSPC', '^VIX'],
    start='2021-01-01',
    end='2026-01-01',
    auto_adjust=True,
)['Close']

# Handle multi-index columns if returned by yfinance
if isinstance(data.columns, pd.MultiIndex):
  data.columns = data.columns.get_level_values(0)

data.columns = ['AAPL', 'SP500', 'VIX']
df = data.dropna()

# --- Feature Engineering ---
df['Return'] = df['AAPL'].pct_change()
df['SP_Return'] = df['SP500'].pct_change()
df['MA_5'] = df['AAPL'].rolling(window=5).mean()
df['MA_20'] = df['AAPL'].rolling(window=20).mean()

# Calculate RSI
delta = df['AAPL'].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
rs = gain.rolling(window=14).mean() / loss.rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + rs))

# Target: 1 if AAPL is higher 3 days from now
df['Target'] = (df['AAPL'].shift(-3) > df['AAPL']).astype(int)
df = df.dropna()

features = ['SP500', 'VIX', 'SP_Return', 'Return', 'MA_5', 'MA_20', 'RSI']
X = df[features]
y = df['Target']

# --- Strict 80:20 Chronological Split ---
train_size = int(len(X) * 0.80)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f'Training Set Size: {len(X_train)} rows')
print(f'Testing Set Size: {len(X_test)} rows')

# --- Train XGBoost Model ---
print('\nTraining XGBoost Model...')
model = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42
)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'\nModel Accuracy: {accuracy * 100:.2f}%')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# --- Plot Results ---
plt.figure(figsize=(10, 4))
df['AAPL'].tail(100).plot(
    title='AAPL Recent Close Prices', color='royalblue', lw=2
)
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
