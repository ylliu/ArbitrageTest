import ccxt
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('TkAgg')  # 切换到 TkAgg 后端
import matplotlib.pyplot as plt
from itertools import combinations

exchange = ccxt.binance({
    'httpsProxy': 'http://127.0.0.1:7890',  # 设置代理
    'timeout': 30000,  # 请求超时时间（可选）
    'tld': 'us',  # Binance US 域名
    'rateLimit': 1200,
    'enableRateLimit': True,
})

# 目标加密货币交易对列表
cryptos = ['AAVE/USDT', 'DOT/USDT', 'DOGE/USDT', 'ARB/USDT', 'AVAX/USDT', 'LINK/USDT',
           'TON/USDT', 'SOL/USDT', 'DYDX/USDT', 'SUI/USDT', 'BNB/USDT', 'LTC/USDT',
           'ORDI/USDT', 'FET/USDT', 'TRX/USDT', 'UNI/USDT', '1000PEPE/USDT',
           '1000SHIB/USDT', 'ALGO/USDT', 'ETH/USDT', 'BTC/USDT', 'XMR/USDT',
           'OP/USDT', 'APT/USDT', 'KAS/USDT', 'ADA/USDT',
           'XRP/USDT', 'XLM/USDT', 'GALA/USDT']

# 时间范围设置（例如，过去一年的数据）
since = exchange.parse8601('2024-01-01T00:00:00Z')  # 开始时间
timeframe = '1d'  # 时间间隔 (每日数据)


# 获取历史数据的函数
def fetch_ohlcv(symbol, since, timeframe):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[['close']]  # 只保留收盘价
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


# 下载所有交易对的数据并合并
data_frames = {}
for crypto in cryptos:
    print(f"Fetching data for {crypto}...")
    df = fetch_ohlcv(crypto, since, timeframe)
    if df is not None:
        data_frames[crypto] = df.rename(columns={'close': crypto.split('/')[0]})
print('fetching data finished')

crypto_pairs = list(combinations(cryptos, 2))

from statsmodels.tsa.stattools import coint

results = []

for pair in crypto_pairs:
    sym1, sym2 = pair

    # Ensure both DataFrames exist in our dictionary
    if sym1 not in data_frames or sym2 not in data_frames:
        print(f"Data not found for one of these symbols: {sym1}, {sym2}")
        continue

    # Extract price series for each symbol

    df1 = pd.Series(data_frames[sym1].values.flatten())  # 将 NumPy 数组转换为 Series
    df2 = pd.Series(data_frames[sym2].values.flatten())  # 将 NumPy 数组转换为 Series

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

# Convert Cointegration Test Result to Dataframe
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


def analyze_pair(data_dict, sym1, sym2):
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

    # # Plot results
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


for cp in cointegrated_pairs:
    sym1 = cp[0]
    sym2 = cp[1]
    print('sym1:', sym1, 'sym2:', sym2)
    pair_results = analyze_pair(data_frames, sym1, sym2)


    def convert_zscore(df, sym1, sym2, window_size=10):
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


    pair_results_df = pair_results['df']

    df_zscore = convert_zscore(pair_results_df, sym1, sym2, window_size=10)

    import numpy as np


    def run_pair_trading(sym1, sym2, data_dict, df_zscore, window_size=10, initial_equity=100_000.0):
        # 1) Align close prices
        df1 = pd.Series(data_frames[sym1].values.flatten()).rename("X")  # 将 NumPy 数组转换为 Series
        df2 = pd.Series(data_frames[sym2].values.flatten()).rename("Y")  # 将 NumPy 数组转换为 Series
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
        df["x_notional"] = 0.02 * initial_equity
        df["y_notional"] = 0.02 * initial_equity

        # Daily PnL for each pair
        df["daily_pnl_x"] = df["x_position"].shift(1) * df["x_notional"] * df["x_return"]
        df["daily_pnl_y"] = df["y_position"].shift(1) * df["y_notional"] * df["y_return"]
        df[["daily_pnl_x", "daily_pnl_y"]] = df[["daily_pnl_x", "daily_pnl_y"]].fillna(0.0)

        df["daily_pnl"] = df["daily_pnl_x"] + df["daily_pnl_y"]
        df["equity"] = initial_equity + df["daily_pnl"].cumsum()

        # 4) Performance metrics
        final_equity = df["equity"].iloc[-1]
        total_return_pct = (final_equity - initial_equity) / initial_equity

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


    pair_trading = run_pair_trading(sym1, sym2, data_frames, df_zscore)


    def plot_return(df):
        # Calculate cumulative return in percentage
        initial_equity = df["equity"].iloc[0]
        df["cumulative_return_pct"] = ((df["equity"] - initial_equity) / initial_equity) * 100
        print(df["cumulative_return_pct"].tail())
        # plt.figure(figsize=(14, 7))
        # plt.plot(df.index, df["cumulative_return_pct"], label="Cumulative Return (%)", color="green", alpha=0.8)
        # plt.title("Cumulative Return in Percentage")
        # plt.xlabel("Date")
        # plt.ylabel("Cumulative Return (%)")
        # plt.legend()
        # plt.grid()
        # plt.show()


    plot_return(pair_trading['df'])
