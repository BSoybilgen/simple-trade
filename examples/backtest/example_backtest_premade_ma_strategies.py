"""Example usage of moving average indicator backtesting strategies."""

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


# ### Adaptive Deviation-Scaled Moving Average (ADS)
params = {"short_window": 25, "long_window": 75}
_run_backtest(data_day, 'ads', params)


# ### Arnaud Legoux Moving Average (ALM)
params = {"short_window": 25, "long_window": 150}
_run_backtest(data_day, 'alm', params)


# ### Adaptive Moving Average (AMA)
params = {"short_window": 10, "long_window": 30}
_run_backtest(data_day, 'ama', params)


# ### Double Exponential Moving Average (DEM)
params = {"short_window": 25, "long_window": 75}
_run_backtest(data_day, 'dem', params)


# ### Exponential Moving Average (EMA)
params = {"short_window": 25, "long_window": 75}
_run_backtest(data_day, 'ema', params)


# ### Elastic Volume Weighted Moving Average (EVW)
params = {"short_window": 10, "long_window": 20}
_run_backtest(data_day, 'evw', params)


# ### Fractal Adaptive Moving Average (FMA)
params = {
    "short_window": 0,
    "long_window": 100,
    "use_adx_filter": True,
    "adx_window": 14,
    "adx_threshold": 25,
}
_run_backtest(data_week, 'fma', params)


# ### Guppy Multiple Moving Average (GMA)
params = {"short_windows": (25, 30), "long_windows": (75, 80)}
_run_backtest(data_day, 'gma', params)


# ### Hull Moving Average (HMA)
params = {"short_window": 25, "long_window": 75}
_run_backtest(data_day, 'hma', params)


# ### Jurik Moving Average (JMA)
params = {"short_length": 6, "long_length": 12}
_run_backtest(data_day, 'jma', params)


# ### Least Squares Moving Average (LSM)
params = {"short_window": 25, "long_window": 75}
_run_backtest(data_day, 'lsm', params)


# ### MESA Adaptive Moving Average (MAM)
params = {"fast_limit": 0.5, "slow_limit": 0.05}
_run_backtest(data_day, 'mam', params)


# ### Simple Moving Average (SMA)
params = {"short_window": 25, "long_window": 75}
_run_backtest(data_day, 'sma', params)


# ### Smoothed Moving Average (SOA)
params = {"short_window": 10, "long_window": 30}
_run_backtest(data_day, 'soa', params)


# ### Sine-Weighted Moving Average (SWM)
params = {"short_window": 10, "long_window": 30}
_run_backtest(data_day, 'swm', params)


# ### Triple Exponential Moving Average (TEM)
params = {"short_window": 50, "long_window": 100}
_run_backtest(data_day, 'tem', params)


# ### Triangular Moving Average (TMA)
params = {"short_window": 10, "long_window": 30}
_run_backtest(data_day, 'tma', params)


# ### Time Series Forecast (TSF)
params = {"window": 14}
_run_backtest(data_day, 'tsf', params)


# ### T3 Moving Average (TT3)
params = {"short_window": 5, "long_window": 10, "v_factor": 0.7}
_run_backtest(data_day, 'tt3', params)


# ### Variable Index Dynamic Average (VID)
params = {"short_window": 14, "long_window": 42, "cmo_window": 9}
_run_backtest(data_day, 'vid', params)


# ### Volume Moving Average (VMA)
params = {"short_window": 10, "long_window": 30}
_run_backtest(data_day, 'vma', params)


# ### Weighted Moving Average (WMA)
params = {"short_window": 25, "long_window": 75}
_run_backtest(data_day, 'wma', params)


# ### Zero-Lag Moving Average (ZMA)
params = {"short_window": 25, "long_window": 75}
_run_backtest(data_day, 'zma', params)