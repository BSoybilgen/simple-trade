"""Example usage of trend indicator backtesting strategies."""

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


# ### Average Directional Index (ADX)
params = {"window": 14, "adx_threshold": 25, "ma_window": 40}
_run_backtest(data_week, 'adx', params)


# ### Aroon Indicator (ARO)
params = {"period": 50}
_run_backtest(data_day, 'aro', params)


# ### Ehlers Adaptive Cyber Cycle (EAC)
params = {"short_alpha": 0.03, "long_alpha": 0}
_run_backtest(data_week, 'eac', params)


# ### Ehlers Instantaneous Trendline (EIT)
params = {"short_alpha": 0.07, "long_alpha": 0.14}
_run_backtest(data_day, 'eit', params)


# ### Hilbert Transform Trendline (HTT)
params = {"short_window": 25, "long_window": 75}
_run_backtest(data_day, 'htt', params)


# ### Ichimoku Cloud (ICH)
params = {"tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52, "displacement": 26}
_run_backtest(data_day, 'ich', params)


# ### McGinley Dynamic (MGD)
params = {"short_window": 10, "long_window": 30}
_run_backtest(data_day, 'mgd', params)


# ### Parabolic SAR (PSA)
params = {"af_initial": 0.003, "af_step": 0.003, "af_max": 0.03}
_run_backtest(data_day, 'psa', params)


# ### Projection Oscillator (PRO)
params = {"window": 14}
_run_backtest(data_day, 'pro', params)


# ### SuperTrend (STR)
params = {"period": 7, "multiplier": 3.0}
_run_backtest(data_day, 'str', params)


# ### TRIX (TRI)
params = {"window": 40}
_run_backtest(data_day, 'tri', params)


# ### Volatility Quality Index (VQI)
params = {"window": 14}
_run_backtest(data_day, 'vqi', params)
