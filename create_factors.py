import pandas as pd
import numpy as np

# ========= CONFIG =========
INPUT_CSV = "data/backtest_data/binanceGlobalAccounts_BTC_binance_5m.csv"
OUTPUT_CSV = "data/backtest_data/binanceGlobalAccounts_BTC_binance_5m_with_factors.csv"
BARS_PER_HOUR = 12   # 5-minute bars => 12 bars per hour
# ==========================


def hours_to_bars(hours: float) -> int:
    return int(hours * BARS_PER_HOUR)


def days_to_bars(days: float) -> int:
    return int(days * 24 * BARS_PER_HOUR)


WINDOWS = {
    "1h": hours_to_bars(1),
    "6h": hours_to_bars(6),
    "1d": days_to_bars(1),
    "3d": days_to_bars(3),
    "7d": days_to_bars(7),
}


def add_time_series_factors(df):
    df = df.copy()
    eps = 1e-8

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)
    trades = df["number_of_trades"].astype(float)

    factors = {}

    # Basic returns
    ret_1 = close.pct_change()
    log_ret = np.log(close).diff()
    factors["ret_1"] = ret_1
    factors["log_ret"] = log_ret

    pos_ret = log_ret.clip(lower=0.0)
    neg_ret = log_ret.clip(upper=0.0)

    for label, n in WINDOWS.items():
        factors[f"logret_sum_{label}"] = log_ret.rolling(n, min_periods=n).sum()
        factors[f"rv_{label}"] = log_ret.rolling(n, min_periods=n).std()
        mean_ret = log_ret.rolling(n, min_periods=n).mean()
        std_ret = log_ret.rolling(n, min_periods=n).std()
        factors[f"trend_sharpe_{label}"] = mean_ret / (std_ret + eps)
        factors[f"upvol_{label}"] = np.sqrt((pos_ret ** 2).rolling(n, min_periods=n).mean())
        factors[f"downvol_{label}"] = np.sqrt((neg_ret ** 2).rolling(n, min_periods=n).mean())
        factors[f"ret_skew_{label}"] = log_ret.rolling(n, min_periods=n).skew()
        factors[f"ret_kurt_{label}"] = log_ret.rolling(n, min_periods=n).kurt()
        factors[f"posret_ratio_{label}"] = (log_ret > 0).rolling(n, min_periods=n).mean()
        factors[f"negret_ratio_{label}"] = (log_ret < 0).rolling(n, min_periods=n).mean()

        ma = close.rolling(n, min_periods=n).mean()
        factors[f"ma_{label}"] = ma
        factors[f"ma_dev_{label}"] = (close - ma) / (ma + eps)
        rolling_min = close.rolling(n, min_periods=n).min()
        rolling_max = close.rolling(n, min_periods=n).max()
        factors[f"price_pos_{label}"] = (close - rolling_min) / (rolling_max - rolling_min + eps)

        vol_ma = vol.rolling(n, min_periods=n).mean()
        vol_std = vol.rolling(n, min_periods=n).std()
        factors[f"vol_rel_{label}"] = vol / (vol_ma + eps)
        factors[f"vol_z_{label}"] = (vol - vol_ma) / (vol_std + eps)

        tr_ma = trades.rolling(n, min_periods=n).mean()
        tr_std = trades.rolling(n, min_periods=n).std()
        factors[f"trades_rel_{label}"] = trades / (tr_ma + eps)
        factors[f"trades_z_{label}"] = (trades - tr_ma) / (tr_std + eps)

        factors[f"corr_ret_vol_{label}"] = log_ret.rolling(n, min_periods=n).corr(vol)
        single_bar_range = np.log(high / (low + eps)).replace([np.inf, -np.inf], 0.0)
        factors[f"range_vol_{label}"] = np.sqrt((single_bar_range ** 2).rolling(n, min_periods=n).mean())

    factors["rev_short_1bar"] = -ret_1
    factors["mom_15m"] = close / close.shift(3) - 1.0
    factors["mom_1h"] = close / close.shift(hours_to_bars(1)) - 1.0

    lag1_ret = log_ret.shift(1)
    for label, n in WINDOWS.items():
        factors[f"autocorr_ret_lag1_{label}"] = log_ret.rolling(n, min_periods=n).corr(lag1_ret)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    factors["true_range"] = true_range
    factors["atr_14bars"] = true_range.rolling(14, min_periods=14).mean()
    factors["atr_1d"] = true_range.rolling(WINDOWS["1d"], min_periods=WINDOWS["1d"]).mean()

    n_bb = 20
    k_bb = 2.0
    ma_bb = close.rolling(n_bb, min_periods=n_bb).mean()
    std_bb = close.rolling(n_bb, min_periods=n_bb).std()
    factors["bb_pos_20bars_2sigma"] = (close - ma_bb) / (k_bb * std_bb + eps)

    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_series = pd.Series(gain, index=df.index)
    loss_series = pd.Series(loss, index=df.index)
    roll_up = gain_series.ewm(alpha=1 / 14, adjust=False).mean()
    roll_down = loss_series.ewm(alpha=1 / 14, adjust=False).mean()
    rs = roll_up / (roll_down + eps)
    factors["rsi_14bars"] = 100.0 - 100.0 / (1.0 + rs)

    for label in ["1d", "3d", "7d"]:
        n = WINDOWS[label]
        rolling_max = close.rolling(n, min_periods=n).max()
        dd = close / (rolling_max + eps) - 1.0
        factors[f"dd_{label}"] = dd
        factors[f"maxdd_{label}"] = dd.rolling(n, min_periods=n).min()

    factors_df = pd.DataFrame(factors, index=df.index)
    out = pd.concat([df, factors_df], axis=1)
    return out


def main():
    df = pd.read_csv(
        INPUT_CSV,
        parse_dates=["open_time", "close_time"],
    ).sort_values("open_time").reset_index(drop=True)
    df_factors = add_time_series_factors(df)
    df_factors = df_factors.dropna().reset_index(drop=True)

    df_factors.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved 5-minute BTCUSDT kline with factors to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


