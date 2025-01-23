import matplotlib
import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
matplotlib.use('TkAgg')  # 切换到 TkAgg 后端
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

# Download stock data for PepsiCo and Coca-Cola
tickers = ['PEP', 'KO']
data = yf.download(tickers, start='2015-01-01', end='2023-01-01')['Close']

# Preview the data
data.head()

# Download stock data for PepsiCo and Coca-Cola
tickers = ['PEP', 'KO']
data = yf.download(tickers, start='2015-01-01', end='2023-01-01')['Close']

# Preview the data
data.head()


# Perform cointegration test
score, p_value, _ = coint(data['PEP'], data['KO'])

print(f'Cointegration test p-value: {p_value}')

# If p-value is low (<0.05), the pairs are cointegrated
if p_value < 0.05:
    print("The pairs are cointegrated.")
else:
    print("The pairs are not cointegrated.")


# Calculate the spread between the two stocks
data['Spread'] = data['PEP'] - data['KO']

# Plot the spread
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Spread'], label='Spread (PEP - KO)')
plt.axhline(data['Spread'].mean(), color='red', linestyle='--', label='Mean')
plt.legend()
plt.title('Spread between PEP and KO')
plt.show()

# Define z-score to normalize the spread
data['Z-Score'] = (data['Spread'] - data['Spread'].mean()) / data['Spread'].std()

# Set thresholds for entering and exiting trades
upper_threshold = 2
lower_threshold = -2

# Initialize signals
data['Position'] = 0

# Generate signals for long and short positions
data['Position'] = np.where(data['Z-Score'] > upper_threshold, -1, data['Position'])  # Short the spread
data['Position'] = np.where(data['Z-Score'] < lower_threshold, 1, data['Position'])   # Long the spread
data['Position'] = np.where((data['Z-Score'] < 1) & (data['Z-Score'] > -1), 0, data['Position'])  # Exit

# Plot z-score and positions
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Z-Score'], label='Z-Score')
plt.axhline(upper_threshold, color='red', linestyle='--', label='Upper Threshold')
plt.axhline(lower_threshold, color='green', linestyle='--', label='Lower Threshold')
plt.legend()
plt.title('Z-Score of the Spread with Trade Signals')
plt.show()

# Calculate daily returns
data['PEP_Return'] = data['PEP'].pct_change()
data['KO_Return'] = data['KO'].pct_change()

# Strategy returns: long spread means buying PEP and shorting KO
data['Strategy_Return'] = data['Position'].shift(1) * (data['PEP_Return'] - data['KO_Return'])

# Cumulative returns
data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()

# Plot cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Cumulative_Return'], label='Cumulative Return from Strategy')
plt.title('Cumulative Returns of Pairs Trading Strategy')
plt.legend()
plt.show()

# Calculate Sharpe Ratio
sharpe_ratio = data['Strategy_Return'].mean() / data['Strategy_Return'].std() * np.sqrt(252)
print(f'Sharpe Ratio: {sharpe_ratio}')

# Calculate max drawdown
cumulative_max = data['Cumulative_Return'].cummax()
drawdown = (cumulative_max - data['Cumulative_Return']) / cumulative_max
max_drawdown = drawdown.max()
print(f'Max Drawdown: {max_drawdown}')