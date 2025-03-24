import ccxt
import pandas as pd
from datetime import datetime
import time


def get_top_volume_symbols(exchange_id='binance', limit=100):
    """
    获取指定交易所交易量靠前的币种

    参数:
    exchange_id (str): 交易所ID，默认为'binance'
    limit (int): 返回的币种数量，默认为100

    返回:
    DataFrame: 包含币种、交易对和24小时交易量的DataFrame
    """
    # 初始化交易所
    try:
        exchange = getattr(ccxt, exchange_id)({
            'httpsProxy': 'http://127.0.0.1:7890',  # 设置代理
            'timeout': 30000,  # 请求超时时间（可选）
            'tld': 'us',  # Binance US 域名
            'enableRateLimit': True,  # 启用请求频率限制
        })
        # exchange = ccxt.binance({
        #     'httpsProxy': 'http://127.0.0.1:7890',  # 设置代理
        #     'timeout': 30000,  # 请求超时时间（可选）
        #     'tld': 'us',  # Binance US 域名
        #     'rateLimit': 1200,
        #     'enableRateLimit': True,
        # })

        # 加载市场
        exchange.load_markets()

        # 获取所有交易对的24小时行情数据
        tickers = exchange.fetch_tickers()

        # 提取需要的数据
        data = []
        for symbol, ticker in tickers.items():
            # 确保有交易量数据
            if 'quoteVolume' in ticker and ticker['quoteVolume'] is not None:
                # 使用计价货币的交易量(通常是USDT, BTC等)
                quote_volume = ticker['quoteVolume']

                # 获取基础货币和计价货币
                base, quote = symbol.split('/')

                data.append({
                    'symbol': symbol,
                    'base_currency': base,
                    'quote_currency': quote,
                    'volume_24h': quote_volume,
                    'last_price': ticker['last'] if 'last' in ticker else None,
                    'percentage_change': ticker['percentage'] if 'percentage' in ticker else None
                })
        print(data)
        # 转换为DataFrame
        df = pd.DataFrame(data)

        # 按交易量降序排序
        df = df.sort_values(by='volume_24h', ascending=False)

        # 返回前N个结果
        return df.head(limit)

    except Exception as e:
        print(f"发生错误: {e}")
        return None


if __name__ == "__main__":
    # 获取币安交易量靠前的100个交易对
    top_symbols = get_top_volume_symbols(exchange_id='binance', limit=100)

    if top_symbols is not None:
        # 格式化输出
        pd.set_option('display.max_rows', None)  # 显示所有行
        pd.set_option('display.width', 1000)  # 设置显示宽度

        print(f"获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n币安交易量靠前的{len(top_symbols)}个交易对:")

        # 格式化交易量数据，使其更易读
        top_symbols['volume_24h_formatted'] = top_symbols['volume_24h'].apply(
            lambda x: f"{x:,.2f}" if x >= 1000000 else f"{x:.2f}"
        )

        # 只显示需要的列
        display_columns = ['symbol', 'base_currency', 'quote_currency', 'volume_24h_formatted', 'last_price']
        print(top_symbols[display_columns].to_string(index=False))

        # 可选：保存到CSV文件
        top_symbols.to_csv(f"binance_top_volume_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)
        print(f"\n数据已保存到CSV文件")