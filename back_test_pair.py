from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import matplotlib
from statsmodels.tsa.stattools import coint
from tqdm import tqdm

matplotlib.use('TkAgg')  # 切换到 TkAgg 后端
import matplotlib.pyplot as plt
from itertools import combinations


class BackTestPair:
    def __init__(self, sym1, sym2):
        self.sym1 = sym1
        self.sym2 = sym2
        self.exchange = ccxt.binance({
            'httpsProxy': 'http://127.0.0.1:7890',  # 设置代理
            'timeout': 30000,  # 请求超时时间（可选）
            'tld': 'us',  # Binance US 域名
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        # 时间范围设置（例如，过去一年的数据）
        self.since = self.exchange.parse8601('2024-01-01T00:00:00Z')  # 开始时间
        self.timeframe = '1h'  # 时间间隔 (每日数据)

    def get_top_50(self):
        try:
            # 获取所有市场数据
            markets = self.exchange.load_markets()
            tickers = self.exchange.fetch_tickers()

            # 筛选出以 USDT 为参考货币的交易对且在2024年1月1日之前上市的
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT'):
                    market = markets.get(symbol, {})
                    # 获取上市日期信息
                    info = market.get('info', {})
                    list_date = None
                    if symbol == 'PNUT/USDT':
                        print(info)

                    # 不同交易所可能有不同的日期字段
                    if 'listDate' in info:
                        list_date = info['listDate']
                    elif 'created_at' in info:
                        list_date = info['created_at']
                    elif 'onboardDate' in info:
                        list_date = info['onboardDate']

                    # 如果找不到日期信息或日期早于2024年1月1日，则添加到列表
                    if list_date is None or (isinstance(list_date, str) and list_date < '2024-01-01'):
                        usdt_pairs.append((symbol, ticker))

            # 按成交量排序
            sorted_pairs = sorted(usdt_pairs, key=lambda item: item[1].get('quoteVolume', 0), reverse=True)

            # 获取成交量前 50 的以 USDT 为参考货币的交易对
            top_50_usdt_pairs = sorted_pairs[:50]
            top_50_usdt_symbols = [pair[0] for pair in top_50_usdt_pairs]

            print("以 USDT 为参考货币，且在2024年1月1日之前上市的成交量前 50 的交易对：", top_50_usdt_symbols)
            return top_50_usdt_symbols

        except Exception as e:
            print(f"发生错误: {e}")
            return []

    # 获取历史数据的函数
    def fetch_ohlcv(self, symbol, since, timeframe):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df[['close']]  # 只保留收盘价
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    def analyze_pair(self, data_dict, sym1, sym2):
        # Extract & align data
        priceX = pd.Series(data_dict[sym1].values.flatten()).rename("X")  # 将 NumPy 数组转换为 Series
        priceY = pd.Series(data_dict[sym2].values.flatten()).rename("Y")  # 将 NumPy 数组转换为 Series
        df = pd.concat([priceX, priceY], axis=1, join="inner").dropna()

        # If insufficient data remains, exit
        if len(df) < 10:
            print(f"Not enough overlapping data for {sym1} & {sym2}. Exiting.")
            return None

        # Prepare & fit OLS model
        X = sm.add_constant(df["X"])
        Y = df["Y"]
        model = sm.OLS(Y, X).fit()

        # Compute predictions & spread
        df["Y_pred"] = model.predict(X)
        df["Spread"] = df["Y"] - df["Y_pred"]

        # Plot results
        # plt.figure(figsize=(10, 5))
        # plt.plot(df.index, df["Y"], label=f"{sym2} (Actual)")
        # plt.plot(df.index, df["Y_pred"], label=f"{sym2} (Predicted from {sym1})")
        # plt.title(f"Pair: {sym1} (X) → {sym2} (Y)")
        # plt.legend()
        # plt.show()
        #
        # plt.figure(figsize=(10, 4))
        # plt.plot(df.index, df["Spread"], label="Spread (Y - Y_pred)")
        # plt.axhline(df["Spread"].mean(), color='red', linestyle='--', label="Spread Mean")
        # plt.title(f"Spread for {sym1} & {sym2}")
        # plt.legend()
        # plt.show()

        results_dict = {
            "model_params": model.params,  # alpha (const) and beta
            "df": df,  # the aligned dataframe with Spread
            "summary": model.summary()  # statsmodels summary object
        }
        return results_dict

    def convert_zscore(self, df, sym1, sym2, window_size=10):
        # Compute rolling mean, rolling std, and Z-score
        df["Spread_MA"] = df["Spread"].rolling(window_size).mean()
        df["Spread_STD"] = df["Spread"].rolling(window_size).std()
        df["Zscore"] = (df["Spread"] - df["Spread_MA"]) / df["Spread_STD"]

        # Visualize the Z-score
        # plt.figure(figsize=(10, 4))
        # plt.plot(df.index, df["Zscore"], label="Z-Score of Spread")
        # plt.axhline(0, color="black", linestyle="--", lw=1)
        # plt.axhline(2.0, color="green", linestyle="--", lw=1, label="+2 Z")
        # plt.axhline(1.0, color="green", linestyle="--", lw=1, label="+1 Z")
        # plt.axhline(-1.0, color="red", linestyle="--", lw=1, label="-1 Z")
        # plt.axhline(-2.0, color="red", linestyle="--", lw=1, label="-2 Z")
        # plt.title(f"Z-Score of Spread (Window={window_size}): {sym1}, {sym2}")
        # plt.legend()
        # plt.show()

        return df

    def get_listing_date(self, symbol):
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            release_date = data.get('genesis_date')
            return release_date
        except requests.RequestException as e:
            print(f"请求出错: {e}")
        except KeyError:
            print("未找到发行日期信息。")

    def run_pair_trading(self, sym1, sym2, data_dict, df_zscore, window_size=10, initial_equity=100_000.0):
        df_sym1 = self.fetch_ohlcv(sym1, self.since, self.timeframe)
        df_sym2 = self.fetch_ohlcv(sym2, self.since, self.timeframe)
        # 1) Align close prices
        df1 = pd.Series(df_sym1.values.flatten()).rename("X")  # 将 NumPy 数组转换为 Series
        df2 = pd.Series(df_sym2.values.flatten()).rename("Y")  # 将 NumPy 数组转换为 Series
        df = pd.concat([df1, df2], axis=1, join="inner").dropna().sort_index()

        if len(df) < window_size:
            return {
                "df": None,
                "metrics": f"Not enough data for {sym1}-{sym2}",
                "trades_df": pd.DataFrame()
            }

        df = df.join(df_zscore, how="inner", rsuffix="_zscore")

        if "Zscore" not in df or df["Zscore"].isna().all():
            return {
                "df": None,
                "metrics": f"Missing or invalid Zscore data for {sym1}-{sym2}",
                "trades_df": pd.DataFrame()
            }

        # 2) Determine positions on each pair
        df["x_position"] = np.nan
        df["y_position"] = np.nan

        # zscore > 2 => Short X, Long Y
        df.loc[df["Zscore"] > 2, ["x_position", "y_position"]] = [-1, 1]

        # -1 < zscore < 1 => Exit
        df.loc[(df["Zscore"] > -1) & (df["Zscore"] < 1), ["x_position", "y_position"]] = [0, 0]

        # zscore < -2 => Long X, Short Y
        df.loc[df["Zscore"] < -2, ["x_position", "y_position"]] = [1, -1]

        # Forward-fill positions in between signals
        df["x_position"] = df["x_position"].ffill().fillna(0)
        df["y_position"] = df["y_position"].ffill().fillna(0)

        # 3) Calculate daily returns from each pair
        df["x_return"] = df["X"].pct_change().fillna(0.0)
        df["y_return"] = df["Y"].pct_change().fillna(0.0)

        # Equity Allocation
        df["x_notional"] = 0.3 * initial_equity
        df["y_notional"] = 0.3 * initial_equity

        # Daily PnL for each pair
        df["daily_pnl_x"] = df["x_position"].shift(1) * df["x_notional"] * df["x_return"]
        df["daily_pnl_y"] = df["y_position"].shift(1) * df["y_notional"] * df["y_return"]
        df[["daily_pnl_x", "daily_pnl_y"]] = df[["daily_pnl_x", "daily_pnl_y"]].fillna(0.0)

        df["daily_pnl"] = df["daily_pnl_x"] + df["daily_pnl_y"]
        df["equity"] = initial_equity + df["daily_pnl"].cumsum()

        # 4) Performance metrics
        final_equity = df["equity"].iloc[-1]
        total_return_pct = round((final_equity - initial_equity) / initial_equity * 100, 2)

        df["equity_return"] = df["equity"].pct_change().fillna(0.0)
        ann_factor = 252
        mean_daily_ret = df["equity_return"].mean()
        std_daily_ret = df["equity_return"].std()

        if std_daily_ret != 0:
            sharpe_ratio = (mean_daily_ret / std_daily_ret) * np.sqrt(ann_factor)
        else:
            sharpe_ratio = np.nan

        neg_returns = df.loc[df["equity_return"] < 0, "equity_return"]
        std_downside = neg_returns.std() if not neg_returns.empty else np.nan
        if std_downside and std_downside != 0:
            sortino_ratio = (mean_daily_ret / std_downside) * np.sqrt(ann_factor)
        else:
            sortino_ratio = np.nan

        df["running_max"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] / df["running_max"]) - 1
        max_drawdown = df["drawdown"].min()

        # 5) Trade-by-trade details
        df["trade_signal"] = (df["x_position"].diff().abs() > 0) | (df["y_position"].diff().abs() > 0)
        trades = df[df["trade_signal"]].copy()
        trades["entry_date"] = trades.index
        trades["exit_date"] = trades["entry_date"].shift(-1)
        trades["pnl"] = trades["daily_pnl"]
        trades["x_position"] = trades["x_position"]
        trades["y_position"] = trades["y_position"]

        trades_df = trades[["entry_date", "exit_date", "x_position", "y_position", "pnl"]]
        num_trades = len(trades_df)
        win_rate = (trades_df[trades_df["pnl"] > 0].shape[0] / num_trades) if num_trades > 0 else np.nan

        metrics = {
            "sym1": sym1,
            "sym2": sym2,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown_pct": max_drawdown,
            "num_trades": num_trades,
            "win_rate_pct": 100.0 * win_rate if not np.isnan(win_rate) else np.nan
        }

        return {
            "df": df,
            "metrics": metrics,
            "trades_df": trades_df
        }

    def run_back_test(self):
        # Fetch historical data for each symbol
        data_dict = {
            self.sym1: self.fetch_ohlcv(self.sym1, self.since, self.timeframe),
            self.sym2: self.fetch_ohlcv(self.sym2, self.since, self.timeframe)
        }

        # Analyze the pair
        results = self.analyze_pair(data_dict, self.sym1, self.sym2)
        if results is None:
            return None

        # Convert Z-score
        df_zscore = self.convert_zscore(results["df"], self.sym1, self.sym2)

        # Run pair trading strategy
        strategy_results = self.run_pair_trading(self.sym1, self.sym2, data_dict, df_zscore)
        return strategy_results

    def get_cointegrated_pairs(self, top_50_symbols):
        results = []
        crypto_pairs = list(combinations(top_50_symbols, 2))
        for pair in tqdm(crypto_pairs, desc="Processing Crypto Pairs"):
            sym1, sym2 = pair
            df_sym1 = self.fetch_ohlcv(sym1, self.since, self.timeframe)
            df_sym2 = self.fetch_ohlcv(sym2, self.since, self.timeframe)
            # 1) Align close prices
            df1 = pd.Series(df_sym1.values.flatten()).rename("X")  # 将 NumPy 数组转换为 Series
            df2 = pd.Series(df_sym2.values.flatten()).rename("Y")  # 将 NumPy 数组转换为 Series
            # Ensure both DataFrames exist in our dictionary
            # Combine into one DataFrame on the same dates (inner join), drop missing values
            combined = pd.concat([df1, df2], axis=1, join="inner").dropna()
            combined.columns = ["Price1", "Price2"]

            # If there's not enough data after alignment, skip
            if len(combined) < 10:
                continue

            # Run Engle-Granger cointegration test
            coint_t, p_value, critical_values = coint(combined["Price1"], combined["Price2"])

            # Check if p-value < 0.05 for significance
            is_significant = (p_value < 0.05)

            # Store results
            results.append({
                "Symbol1": sym1,
                "Symbol2": sym2,
                "Test Statistic": coint_t,
                "p-value": p_value,
                "5% Critical Value": critical_values[0],  # 1%, 5%, 10% in array
                "Is_Cointegrated_5pct": is_significant
            })

        for res in results:
            status = "Cointegrated" if res["Is_Cointegrated_5pct"] else "Not Cointegrated"
            print(
                f"{res['Symbol1']} & {res['Symbol2']} | "
                f"Test Statistic: {res['Test Statistic']:.3f} | "
                f"p-value: {res['p-value']:.3f} | "
                f"5% Crit. Value: {res['5% Critical Value']:.3f} | "
                f"Result: {status}"
            )

        results_df = pd.DataFrame(results)
        # Filter for rows where Is_Cointegrated_5pct is True
        filtered_df = results_df[results_df['Is_Cointegrated_5pct'] == True]

        # Create a new list of only the cointegrated pairs
        cointegrated_pairs = [
            (row['Symbol1'], row['Symbol2'])
            for _, row in filtered_df.iterrows()
        ]

        print("Cointegrated Pairs (5% level):")
        for cp in cointegrated_pairs:
            print(cp)
        print('Cointegrated size:', len(cointegrated_pairs))
        return cointegrated_pairs

    def run_top_50_back_test(self):
        top_50_symbols = self.get_top_50()
        print(top_50_symbols)
        cointegrated_pairs = self.get_cointegrated_pairs(top_50_symbols)
        print(cointegrated_pairs)


if __name__ == '__main__':
    back_test = BackTestPair('BTC/USDT', 'ETH/USDT')
    top_50_symbols = back_test.get_top_50()
    res = back_test.get_cointegrated_pairs(top_50_symbols)
    print(res)
    # data = [
    #     ('USDC/USDT', 'TRUMP/USDT'),
    #     ('USDC/USDT', 'ORCA/USDT'),
    #     ('SOL/USDT', 'TRUMP/USDT'),
    #     ('SOL/USDT', 'S/USDT'),
    #     ('SOL/USDT', 'CAKE/USDT'),
    #     ('SOL/USDT', '1000SATS/USDT'),
    #     ('SOL/USDT', 'HBAR/USDT'),
    #     ('SOL/USDT', 'TAO/USDT'),
    #     ('SOL/USDT', 'ORDI/USDT'),
    #     ('SOL/USDT', 'WLD/USDT'),
    #     ('SOL/USDT', 'JUP/USDT'),
    #     ('TRUMP/USDT', 'XRP/USDT'),
    #     ('TRUMP/USDT', '1000SATS/USDT'),
    #     ('TRUMP/USDT', 'HBAR/USDT'),
    #     ('TRUMP/USDT', 'DF/USDT'),
    #     ('TRUMP/USDT', 'RUNE/USDT'),
    #     ('TRUMP/USDT', 'NEAR/USDT'),
    #     ('XRP/USDT', 'PEPE/USDT'),
    #     ('XRP/USDT', 'AVAX/USDT'),
    #     ('XRP/USDT', 'HBAR/USDT'),
    #     ('XRP/USDT', 'AAVE/USDT'),
    #     ('XRP/USDT', 'DOT/USDT'),
    #     ('BNX/USDT', 'AUCTION/USDT'),
    #     ('BNX/USDT', 'ENA/USDT'),
    #     ('BNX/USDT', 'LAYER/USDT'),
    #     ('PNUT/USDT', 'TRX/USDT'),
    #     ('PNUT/USDT', 'SUI/USDT'),
    #     ('PNUT/USDT', 'LTC/USDT'),
    #     ('PNUT/USDT', 'LINK/USDT'),
    #     ('PNUT/USDT', 'KAITO/USDT'),
    #     ('AUCTION/USDT', 'LAYER/USDT'),
    #     ('BNB/USDT', 'ENA/USDT'),
    #     ('BNB/USDT', 'FORM/USDT'),
    #     ('BNB/USDT', 'LAYER/USDT'),
    #     ('BNB/USDT', 'KAITO/USDT'),
    #     ('PEPE/USDT', 'ADA/USDT'),
    #     ('PEPE/USDT', 'AVAX/USDT'),
    #     ('PEPE/USDT', 'AAVE/USDT'),
    #     ('PEPE/USDT', 'DOT/USDT'),
    #     ('ADA/USDT', 'AVAX/USDT'),
    #     ('ADA/USDT', 'AAVE/USDT'),
    #     ('W/USDT', 'HBAR/USDT'),
    #     ('W/USDT', 'ZRO/USDT'),
    #     ('W/USDT', 'RUNE/USDT'),
    #     ('W/USDT', 'JUP/USDT'),
    #     ('S/USDT', '1000SATS/USDT'),
    #     ('S/USDT', 'RUNE/USDT'),
    #     ('S/USDT', 'JUP/USDT'),
    #     ('WIF/USDT', 'OM/USDT'),
    #     ('WIF/USDT', 'NEIRO/USDT'),
    #     ('WIF/USDT', 'JUP/USDT'),
    #     ('AVAX/USDT', 'TAO/USDT'),
    #     ('AVAX/USDT', 'DOT/USDT'),
    #     ('AVAX/USDT', 'ORDI/USDT'),
    #     ('ENA/USDT', 'FORM/USDT'),
    #     ('ENA/USDT', 'LINK/USDT'),
    #     ('ENA/USDT', 'DF/USDT'),
    #     ('ENA/USDT', 'KAITO/USDT'),
    #     ('CAKE/USDT', '1000SATS/USDT'),
    #     ('CAKE/USDT', 'HBAR/USDT'),
    #     ('CAKE/USDT', 'CRV/USDT'),
    #     ('CAKE/USDT', 'ANKR/USDT'),
    #     ('CAKE/USDT', 'TAO/USDT'),
    #     ('CAKE/USDT', 'NEAR/USDT'),
    #     ('CAKE/USDT', 'ORDI/USDT'),
    #     ('CAKE/USDT', 'WLD/USDT'),
    #     ('FORM/USDT', 'LINK/USDT'),
    #     ('FORM/USDT', 'CRV/USDT'),
    #     ('FORM/USDT', 'AAVE/USDT'),
    #     ('FORM/USDT', 'ANKR/USDT'),
    #     ('FORM/USDT', 'DF/USDT'),
    #     ('FORM/USDT', 'ZRO/USDT'),
    #     ('FORM/USDT', 'TAO/USDT'),
    #     ('FORM/USDT', 'LAYER/USDT'),
    #     ('FORM/USDT', 'OM/USDT'),
    #     ('FORM/USDT', 'DOT/USDT'),
    #     ('FORM/USDT', 'NEIRO/USDT'),
    #     ('FORM/USDT', 'WLD/USDT'),
    #     ('FORM/USDT', 'EOS/USDT'),
    #     ('FORM/USDT', 'TIA/USDT'),
    #     ('1000SATS/USDT', 'WLD/USDT'),
    #     ('HBAR/USDT', 'CRV/USDT'),
    #     ('HBAR/USDT', 'ANKR/USDT'),
    #     ('HBAR/USDT', 'TAO/USDT'),
    #     ('HBAR/USDT', 'ORDI/USDT'),
    #     ('HBAR/USDT', 'WLD/USDT'),
    #     ('CRV/USDT', 'TAO/USDT'),
    #     ('CRV/USDT', 'WLD/USDT'),
    #     ('ORCA/USDT', 'RUNE/USDT'),
    #     ('ORCA/USDT', 'TIA/USDT'),
    #     ('AAVE/USDT', 'DOT/USDT'),
    #     ('ANKR/USDT', 'WLD/USDT'),
    #     ('ANKR/USDT', 'JUP/USDT'),
    #     ('DF/USDT', 'ORDI/USDT'),
    #     ('DF/USDT', 'WLD/USDT'),
    #     ('DF/USDT', 'EOS/USDT'),
    #     ('TAO/USDT', 'RUNE/USDT'),
    #     ('TAO/USDT', 'KAITO/USDT'),
    #     ('TAO/USDT', 'OM/USDT'),
    #     ('TAO/USDT', 'NEAR/USDT'),
    #     ('LAYER/USDT', 'RUNE/USDT'),
    #     ('LAYER/USDT', 'KAITO/USDT'),
    #     ('LAYER/USDT', 'BEAMX/USDT'),
    #     ('LAYER/USDT', 'OM/USDT'),
    #     ('LAYER/USDT', 'DOT/USDT'),
    #     ('LAYER/USDT', 'NEAR/USDT'),
    #     ('LAYER/USDT', 'ORDI/USDT'),
    #     ('LAYER/USDT', 'TIA/USDT'),
    #     ('RUNE/USDT', 'NEAR/USDT'),
    #     ('BEAMX/USDT', 'DOT/USDT'),
    #     ('BEAMX/USDT', 'WLD/USDT'),
    #     ('BEAMX/USDT', 'EOS/USDT'),
    #     ('BEAMX/USDT', 'TIA/USDT'),
    #     ('BEAMX/USDT', 'JUP/USDT'),
    #     ('NEIRO/USDT', 'JUP/USDT'),
    #     ('ORDI/USDT', 'WLD/USDT'),
    #     ('ORDI/USDT', 'EOS/USDT'),
    #     ('ORDI/USDT', 'TIA/USDT'),
    #     ('ORDI/USDT', 'JUP/USDT')
    # ]
    # coinegrated_pair = [('USDC/USDT', 'TRUMP/USDT'), ('SOL/USDT', 'TRUMP/USDT'), ('SOL/USDT', 'S/USDT'),
    #                     ('SOL/USDT', 'CAKE/USDT'), ('TRUMP/USDT', 'XRP/USDT'), ('XRP/USDT', 'PEPE/USDT'),
    #                     ('BNX/USDT', 'AUCTION/USDT'), ('PNUT/USDT', 'TRX/USDT'), ('PNUT/USDT', 'SUI/USDT'),
    # #                     ('PEPE/USDT', 'ADA/USDT')]
    results = []
    for pair in res:
        sym1 = pair[0]
        sym2 = pair[1]
        back_test = BackTestPair(sym1, sym2)
        res = back_test.run_back_test()
        results.append(res)
        print(res['metrics']['sym1'], res['metrics']['sym2'], res['metrics']['total_return_pct'])
        #PNUT/USDT LTC/USDT 30.53
