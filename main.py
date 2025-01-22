import ccxt
import pandas as pd

# 初始化交易所
# 初始化 Binance 交易所，配置代理和顶级域名
exchange = ccxt.binance({
    'httpsProxy': 'http://127.0.0.1:7890',  # 设置代理
    'timeout': 30000,  # 请求超时时间（可选）
    'tld': 'us',  # Binance US 域名
})

# 定义获取历史数据的函数
def fetch_data(symbol, timeframe, start, end):
    since = exchange.parse8601(start)  # 转换起始时间
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # 过滤数据到指定结束日期
    end_date = pd.to_datetime(end)
    df = df[df.index <= end_date]
    return df
# 获取 SOL/USDT 和 JUP/USDT 的数据
symbol1 = 'SOL/USDT'
symbol2 = 'JUP/USDT'
timeframe = '1h'  # 数据间隔
start = '2024-02-01 00:00:00'
end = '2024-09-30 23:59:59'

# 下载数据
data1 = fetch_data(symbol1, timeframe, start, end)
data2 = fetch_data(symbol2, timeframe, start, end)
print(data1)

