"""Example usage of pre-made statistics indicator backtesting strategies."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import download_data, run_premade_trade, print_results

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# --- Global Parameters ---
GLOBAL_PARAMS = {
    'initial_cash': 10000,
    'commission_long': 0.001,
    'commission_short': 0.001,
    'short_borrow_fee_inc_rate': 0.0,
    'long_borrow_fee_inc_rate': 0.0,
    'trading_type': 'long',
    'day1_position': 'none',
    'risk_free_rate': 0.0,
    'fig_control': 1,
}


def _fetch_data(symbol: str, start: str, end: str, interval: str = '1d'):
    print(f"\nDownloading data for {symbol} ({interval})...")
    return download_data(symbol, start, end, interval=interval)


def _maybe_show(fig):
    if fig is not None:
        plt.show(block=True)
        plt.close(fig)


def _run_backtest(data, strategy: str, params: dict):
    all_params = {**GLOBAL_PARAMS, **params}
    results, portfolio, fig = run_premade_trade(data, strategy, all_params)
    print_results(results)
    _maybe_show(fig)


# Fetch data once
data_day = _fetch_data('AAPL', '2020-01-01', '2022-12-31', '1d')
data_week = _fetch_data('AAPL', '2020-01-01', '2024-12-31', '1wk')


# ### Kurtosis (KUR)
params = {"window": 30, "upper_pct": 85, "lower_pct": 15, "lookback": 150}
_run_backtest(data_week, 'kur', params)


# ### Mean Absolute Deviation (MAB)
params = {"window": 20, "upper_pct": 80, "lower_pct": 20, "lookback": 120}
_run_backtest(data_day, 'mab', params)


# ### Median (MED)
params = {"window": 25}
_run_backtest(data_day, 'med', params)


# ### Quartiles (QUA)
params = {"window": 20, "upper_quantile": 0.75, "lower_quantile": 0.25}
_run_backtest(data_day, 'qua', params)


# ### Skewness (SKW)
params = {"window": 40, "upper_pct": 85, "lower_pct": 15, "lookback": 160}
_run_backtest(data_week, 'skw', params)


# ### Standard Deviation (STD)
params = {"window": 20, "upper_pct": 80, "lower_pct": 20, "lookback": 120}
_run_backtest(data_day, 'std', params)


# ### Variance (VAR)
params = {"window": 20, "upper_pct": 80, "lower_pct": 20, "lookback": 120}
_run_backtest(data_day, 'var', params)


# ### Z-Score (ZSC)
params = {"window": 20, "upper_threshold": 2.0, "lower_threshold": -2.0}
_run_backtest(data_day, 'zsc', params)
