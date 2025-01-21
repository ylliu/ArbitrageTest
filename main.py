import ccxt
import pandas as pd

# 初始化交易所
# 初始化 Binance 交易所，配置代理和顶级域名
exchange = ccxt.binance({
    'httpsProxy': 'http://127.0.0.1:7890',  # 设置代理
    'timeout': 30000,                      # 请求超时时间（可选）
    'tld': 'us',                           # Binance US 域名
})

# 定义获取历史数据的函数
def fetch_data(symbol, timeframe, since):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


# 获取 SOL/USDT 和 JUP/USDT 的数据
symbol1 = 'SOL/USDT'
symbol2 = 'JUP/USDT'
timeframe = '1h'  # 数据间隔
since = exchange.parse8601('2025-01-01T00:00:00Z')

data_sol = fetch_data(symbol1, timeframe, since)
data_jup = fetch_data(symbol2, timeframe, since)

# 合并数据
data = pd.merge(data_sol['close'], data_jup['close'], left_index=True, right_index=True, suffixes=('_SOL', '_JUP'))

print(data.head(5))
import numpy as np

# 计算价差
data['Spread'] = data['close_SOL'] - data['close_JUP']

# 计算滚动均值和标准差
window = 20
data['Spread_Mean'] = data['Spread'].rolling(window=window).mean()
data['Spread_Std'] = data['Spread'].rolling(window=window).std()

# 设置上下阈值
entry_z = 2  # 进入交易的 Z 分数
exit_z = 0  # 退出交易的 Z 分数
data['Upper_Threshold'] = data['Spread_Mean'] + entry_z * data['Spread_Std']
data['Lower_Threshold'] = data['Spread_Mean'] - entry_z * data['Spread_Std']
data['Exit_Threshold'] = data['Spread_Mean'] + exit_z * data['Spread_Std']

# 生成交易信号
data['Position'] = 0
data.loc[data['Spread'] > data['Upper_Threshold'], 'Position'] = -1  # 做空 SOL，做多 JUP
data.loc[data['Spread'] < data['Lower_Threshold'], 'Position'] = 1  # 做多 SOL，做空 JUP
data.loc[(data['Spread'] * data['Position'] < data['Exit_Threshold']), 'Position'] = 0  # 平仓

# 计算每日收益率
data['SOL_Returns'] = data['close_SOL'].pct_change()
data['JUP_Returns'] = data['close_JUP'].pct_change()
data['Returns'] = (data['SOL_Returns'] - data['JUP_Returns']) * data['Position'].shift(1)

# 计算累计收益
data['Cumulative_Returns'] = (1 + data['Returns']).cumprod()

# 绘制结果
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))

plt.subplot(211)
plt.plot(data['close_SOL'], label='SOL/USDT')
plt.plot(data['close_JUP'], label='JUP/USDT')
plt.title('Asset Prices')
plt.legend()

plt.subplot(212)
plt.plot(data['Spread'], label='Spread')
plt.plot(data['Spread_Mean'], label='Spread Mean')
plt.fill_between(data.index, data['Upper_Threshold'], data['Lower_Threshold'], color='gray', alpha=0.3,
                 label='Entry Zone')
plt.plot(data.index, data['Position'] * 10, label='Trading Signal', color='magenta', marker='o', linestyle='None')
plt.title('Spread and Trading Signals')
plt.legend()

plt.tight_layout()
plt.show()

# 打印策略表现
total_return = data['Cumulative_Returns'].iloc[-1] - 1
print(f"Total Return: {total_return:.2%}")
