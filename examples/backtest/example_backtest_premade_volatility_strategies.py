"""Example usage of volatility indicator backtesting strategies."""

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
data_week = _fetch_data('AAPL', '2020-01-01', '2022-12-31', '1wk')


# ### Acceleration Bands (ACB)
params = {"period": 20, "factor": 0.001}
_run_backtest(data_day, 'acb', params)


# ### Average True Range Percent (ATP)
params = {"window": 14, "upper": 4.0, "lower": 2.0}
_run_backtest(data_day, 'atp', params)


# ### Average True Range (ATR)
params = {"window": 14, "upper_pct": 80, "lower_pct": 20, "lookback": 100}
_run_backtest(data_day, 'atr', params)


# ### Bollinger Band Width (BBW)
params = {"window": 20, "num_std": 2.0, "upper": 10.0, "lower": 4.0}
_run_backtest(data_day, 'bbw', params)


# ### Bollinger Bands (BOL)
params = {"window": 20, "num_std": 2}
_run_backtest(data_day, 'bol', params)


# ### Chaikin Volatility (CHA)
params = {"ema_window": 10, "roc_window": 10, "upper": 20.0, "lower": -20.0}
_run_backtest(data_day, 'cha', params)


# ### Choppiness Index (CHO)
params = {"period": 14, "upper": 61.8, "lower": 38.2}
_run_backtest(data_day, 'cho', params)


# ### Donchian Channels (DON)
params = {"window": 20}
_run_backtest(data_day, 'don', params)


# ### Dynamic Volatility Index (DVI)
params = {"magnitude_period": 5, "stretch_period": 100, "smooth_period": 5, "upper": 70, "lower": 30}
_run_backtest(data_day, 'dvi', params)


# ### Efficiency Ratio (EFR)
params = {"period": 10, "upper": 0.7, "lower": 0.3}
_run_backtest(data_day, 'efr', params)


# ### Fractal Dimension Index (FDI)
params = {"period": 20, "upper": 1.6, "lower": 1.4}
_run_backtest(data_day, 'fdi', params)


# ### Garman-Klass Volatility (GRV)
params = {"period": 20, "upper": 30.0, "lower": 15.0}
_run_backtest(data_day, 'grv', params)


# ### Heikin-Ashi Volatility (HAV)
params = {"period": 14, "method": 'atr', "upper_pct": 80, "lower_pct": 20, "lookback": 100}
_run_backtest(data_day, 'hav', params)


# ### Historical Volatility (HIV)
params = {"period": 20, "upper": 40.0, "lower": 20.0}
_run_backtest(data_day, 'hiv', params)


# ### Keltner Channels (KEL)
params = {"ema_window": 20, "atr_window": 10, "atr_multiplier": 2.0}
_run_backtest(data_day, 'kel', params)


# ### Median Absolute Deviation (MAD)
params = {"period": 20, "upper": 3.0, "lower": 1.5}
_run_backtest(data_day, 'mad', params)


# ### Mass Index (MAI)
params = {"ema_period": 9, "sum_period": 25, "upper": 26.0, "lower": 24.5}
_run_backtest(data_day, 'mai', params)


# ### Normalized ATR (NAT)
params = {"window": 14, "upper": 3.5, "lower": 2.0}
_run_backtest(data_day, 'nat', params)


# ### Parkinson Volatility (PAV)
params = {"period": 20, "upper": 30.0, "lower": 20.0}
_run_backtest(data_day, 'pav', params)


# ### Price Channel Width (PCW)
params = {"period": 20, "upper": 17.5, "lower": 10.0}
_run_backtest(data_day, 'pcw', params)


# ### Rogers-Satchell Volatility (RSV)
params = {"period": 20, "upper": 30.0, "lower": 20.0}
_run_backtest(data_day, 'rsv', params)


# ### Relative Volatility Index (RVI)
params = {"window": 10, "rvi_period": 14, "upper": 70, "lower": 30}
_run_backtest(data_day, 'rvi', params)


# ### Stochastic Volatility Index (SVI)
params = {"atr_period": 14, "stoch_period": 14, "smooth_k": 3, "smooth_d": 3, "upper": 80, "lower": 20}
_run_backtest(data_day, 'svi', params)


# ### TSI Volatility (TSV)
params = {"atr_period": 14, "long_period": 25, "short_period": 13}
_run_backtest(data_day, 'tsv', params)


# ### Ulcer Index (ULI)
params = {"period": 14, "upper": 5.0, "lower": 1.0}
_run_backtest(data_day, 'uli', params)


# ### Vertical Horizontal Filter (VHF)
params = {"period": 28, "upper": 0.40, "lower": 0.25}
_run_backtest(data_day, 'vhf', params)


# ### Volatility Ratio (VRA)
params = {"short_period": 5, "long_period": 20, "upper": 1.25, "lower": 0.8}
_run_backtest(data_day, 'vra', params)


# ### Volatility Switch Index (VSI)
params = {"short_period": 10, "long_period": 50, "threshold": 1.2, "upper": 0.5, "lower": 0.5}
_run_backtest(data_day, 'vsi', params)
