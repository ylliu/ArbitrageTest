import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

# Download stock data for PepsiCo and Coca-Cola
tickers = ['PEP', 'KO']
data = yf.download(tickers, start='2015-01-01', end='2023-01-01')['Adj Close']

# Preview the data
data.head()

