import vectorbt as vbt
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, Callable, List, Tuple
from urllib.parse import urlunparse
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# ======================
# DATA PREPROCESSING
# ======================
class DataPreprocessor:
    @staticmethod
    def normalize_alpha(
        df: pd.DataFrame,
        alpha_col: str,
        methods: List[str],
        window: int = 100
    ) -> Dict[str, pd.Series]:
        """
        Proper normalization with data leakage prevention (only past data).

        Returns a dict: method_name -> normalized Series (same index as df[alpha_col]).
        """
        normalized = {}
        raw_series = df[alpha_col].dropna()

        for method in methods:
            if method == 'zscore':
                rolling_mean = raw_series.rolling(window=window, min_periods=window).mean()
                rolling_std = raw_series.rolling(window=window, min_periods=window).std(ddof=1)
                values = (raw_series - rolling_mean) / rolling_std

            elif method == 'zscore_pop':
                rolling_mean = raw_series.rolling(window=window, min_periods=window).mean()
                rolling_std = raw_series.rolling(window=window, min_periods=window).std(ddof=0)
                values = (raw_series - rolling_mean) / rolling_std

            elif method == 'minmax':
                rolling_min = raw_series.rolling(window=window, min_periods=window).min()
                rolling_max = raw_series.rolling(window=window, min_periods=window).max()
                values = 2 * (raw_series - rolling_min) / (rolling_max - rolling_min) - 1

            elif method == 'robust':
                rolling_median = raw_series.rolling(window=window, min_periods=window).median()
                rolling_iqr = raw_series.rolling(window=window, min_periods=window).apply(
                    lambda x: np.percentile(x, 75) - np.percentile(x, 25)
                )
                values = (raw_series - rolling_median) / rolling_iqr

            elif method == 'mom':
                # momentum: log(price / price_n_bars_ago)
                values = np.log(raw_series / raw_series.shift(window))

            elif method == 'ma':
                # deviation from moving average in %
                rolling_mean = raw_series.rolling(window=window, min_periods=window).mean()
                values = raw_series / rolling_mean - 1

            elif method == 'ma_diff':
                # difference from moving average in absolute terms
                rolling_mean = raw_series.rolling(window=window, min_periods=window).mean()
                values = raw_series - rolling_mean

            else:
                continue

            normalized[method] = pd.Series(
                values,
                index=raw_series.index,
                name=f"{alpha_col}_{method}"
            )
        return normalized


class StrategyFactory:
    @staticmethod
    def create_strategy(strategy_type: str, lookback_window: int = 100) -> Callable:
        # kept here in case you later need lookback_window for something
        def _rolling_mean(a: pd.Series, window: int) -> pd.Series:
            return a.rolling(window=window, min_periods=window).mean().bfill()

        def SM_a(c: pd.Series, a: pd.Series, t: float, e: float) -> Tuple[pd.Series, ...]:
            short_entries = (a > (t)).astype(bool)
            short_exits = (a < (t)).astype(bool)
            long_entries = (a < (e)).astype(bool)
            long_exits = (a > (e)).astype(bool)
            return long_entries, long_exits, short_entries, short_exits

        def LM(c: pd.Series, a: pd.Series, t: float, e: float) -> Tuple[pd.Series, ...]:
            long_entries = (a > (t)).astype(bool)
            long_exits = (a < (e)).astype(bool)
            short_entries = (a < (-t)).astype(bool)
            short_exits = (a > (-e)).astype(bool)
            return long_entries, long_exits, short_entries, short_exits

        def SM(c: pd.Series, a: pd.Series, t: float, e: float) -> Tuple[pd.Series, ...]:
            long_entries = (a < (-t)).astype(bool)
            long_exits = (a > (-e)).astype(bool)
            short_entries = (a > (+t)).astype(bool)
            short_exits = (a < (+e)).astype(bool)
            return long_entries, long_exits, short_entries, short_exits

        strategies = {
            'SM_a': SM_a,
            'LM': LM,
            'SM': SM,
        }

        if strategy_type not in strategies:
            raise ValueError(f"Strategy '{strategy_type}' not recognized")
        return strategies[strategy_type]


# ======================
# BACKTESTING ENGINE
# ======================
class Backtester:
    def __init__(self, close: pd.Series, alpha: pd.Series):
        self.close = close
        self.alpha = alpha

    def run_backtest(self, strategy: Callable, strategy_params: dict) -> vbt.Portfolio:
        long_entries, long_exits, short_entries, short_exits = strategy(
            self.close,
            self.alpha,
            strategy_params['threshold'],
            strategy_params['exit_threshold']
        )

        return vbt.Portfolio.from_signals(
            self.close,
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            # init_cash=100000,
            # size=1000,
            # size_type='value',
            # upon_opposite_entry='close',
            freq='5m',
            slippage=0.0002,
            fees=0.0003,
        )

    @staticmethod
    def analyze_results(
        portfolio: vbt.Portfolio,
        strategy_name: str,
        norm_method: str,
        threshold: float,
        exit_threshold: float,
        plot: bool = True
    ) -> Dict[str, float]:
        """
        Print stats, compute APY, optionally plot, and return key metrics.
        """
        stats = portfolio.stats()

        total_return = float(stats.get('Total Return [%]', 0))
        total_period = stats.get('Period', pd.Timedelta(0))

        # Ensure total_period is Timedelta
        if not isinstance(total_period, pd.Timedelta):
            try:
                total_period = pd.to_timedelta(total_period)
            except Exception:
                total_period = pd.Timedelta(0)

        total_days = total_period.days if total_period.days > 0 else 1
        apy = (1 + total_return / 100) ** (365 / total_days) - 1
        apy_percentage = apy * 100  # Convert to percentage

        print(stats)
        print("============== BACKTEST METRICS ==============")
        print(f"Strategy: {strategy_name}")
        print(f"Normalization: {norm_method}")
        print(f"Threshold (offset): {threshold}")
        print(f"Exit threshold (offset): {exit_threshold}")
        print(f"APY [%]: {apy_percentage:.2f}")
        print("----------------------------------------------")

        if plot:
            portfolio.plot().show()

        return {
            'sharpe': float(stats.get('Sharpe Ratio', 0)),
            'total_return': total_return,
            'max_drawdown': float(stats.get('Max Drawdown [%]', 0))
        }

    # ========= NEW: trade labelling =========
    @staticmethod
    def label_trades(
        portfolio: vbt.Portfolio,
        label_name: str = "trade_ret_pct",
    ) -> pd.Series:
        """
        Create a label series aligned with portfolio index:

        - At the entry bar of each trade, store the trade's percentage return
          (either from vectorbt's 'return' column, or pnl/notional fallback)
        - NaN everywhere else.
        """
        trades = portfolio.trades
        recs = trades.records
        if not isinstance(recs, pd.DataFrame):
            recs = recs.to_df()

        index = portfolio.wrapper.index
        labels = pd.Series(np.nan, index=index, name=label_name)

        # If vectorbt already computed per-trade return, use it
        has_return_col = "return" in recs.columns

        for _, row in recs.iterrows():
            entry_idx = int(row["entry_idx"])

            if has_return_col:
                # vectorbt's own trade return (usually decimal, e.g. 0.05 = +5%)
                trade_ret = float(row["return"])
            else:
                # Fallback: compute % return from pnl and entry notional
                pnl = float(row["pnl"])

                # These column names are standard; adjust if your version differs.
                entry_price = float(row.get("entry_price", np.nan))
                size = float(row.get("size", np.nan))

                notional = abs(entry_price * size)
                if notional == 0 or np.isnan(notional):
                    # Can't compute a meaningful percentage
                    continue

                trade_ret = pnl / notional  # e.g. 0.05 = +5%

            labels.iloc[entry_idx] = trade_ret

        return labels


def create_file_uri(path: str) -> str:
    """Create proper file URI for macOS"""
    return urlunparse(('file', '', str(Path(path).absolute()), '', '', ''))


if __name__ == "__main__":
    input_file = "data/backtest_data/binanceGlobalAccounts_BTC_binance_5m_with_factors.csv"  # <- put your actual CSV here
    alpha_col = "longPct"  # <- your alpha column name here
    strategies = ['SM']  # Choose a specific strategy
    normalization_methods = ['minmax']  # Choose a specific normalization method
    threshold = 0.7894  # Define a specific threshold
    exit_threshold = 0.578  # Define a specific exit threshold
    lookback_window = 1450

    # Load and preprocess data
    df = pd.read_csv(input_file, index_col='open_time', parse_dates=True)

    close = df['close']
    normalized_alphas = DataPreprocessor.normalize_alpha(
        df,
        alpha_col,
        normalization_methods,
        lookback_window
    )

    # Data validation
    print("=== Normalization Verification ===")
    for method, series in normalized_alphas.items():
        print(f"\n{method}:")
        print(f"Mean: {series.mean():.4f} | Std: {series.std():.4f}")
        print(f"Min: {series.min():.4f} | Max: {series.max():.4f}")

    results = []

    for norm_method, norm_alpha in normalized_alphas.items():
        for strategy_name in strategies:
            strategy_func = StrategyFactory.create_strategy(strategy_name, lookback_window)

            backtester = Backtester(close, norm_alpha)

            # Run backtest
            portfolio = backtester.run_backtest(
                strategy_func,
                {'threshold': threshold, 'exit_threshold': exit_threshold}
            )

            # Analyze performance
            metrics = Backtester.analyze_results(
                portfolio,
                strategy_name=strategy_name,
                norm_method=norm_method,
                threshold=threshold,
                exit_threshold=exit_threshold,
                plot=True
            )

            # ===== NEW: create label column for ML =====
            label_col_name = f"label_{strategy_name}_{norm_method}"
            trade_labels = Backtester.label_trades(portfolio, label_col_name)
            df[label_col_name] = trade_labels

            results.append({
                'strategy': strategy_name,
                'norm_method': norm_method,
                'threshold': threshold,
                'exit_threshold': exit_threshold,
                'sharpe': metrics['sharpe'],
                'total_return': metrics['total_return'],
                'max_drawdown': metrics['max_drawdown'],
            })

    # ===== NEW: save df with labels for future ML training =====
    output_path = Path(input_file).with_name(Path(input_file).stem + "_with_labels.csv")
    df.to_csv(output_path)
    print(f"\nSaved labeled data to: {output_path}")

