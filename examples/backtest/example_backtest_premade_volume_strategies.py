"""Example usage of volume indicator backtesting strategies."""

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


# ### Accumulation/Distribution Line (ADL)
params = {"sma_period": 20}
_run_backtest(data_day, 'adl', params)


# ### Accumulation/Distribution Oscillator (ADO)
params = {"period": 14}
_run_backtest(data_day, 'ado', params)


# ### Bill Williams Market Facilitation Index (BWM)
params = {"upper_pct": 80, "lower_pct": 20, "lookback": 100}
_run_backtest(data_day, 'bwm', params)


# ### Chaikin Money Flow (CMF)
params = {"period": 20}
_run_backtest(data_day, 'cmf', params)


# ### Ease of Movement (EMV)
params = {"period": 14}
_run_backtest(data_day, 'emv', params)


# ### Force Index (FOI)
params = {"period": 13}
_run_backtest(data_day, 'foi', params)


# ### Finite Volume Elements (FVE)
params = {"period": 22, "factor": 0.3}
_run_backtest(data_day, 'fve', params)


# ### Klinger Volume Oscillator (KVO)
params = {"fast_period": 34, "slow_period": 55, "signal_period": 13}
_run_backtest(data_day, 'kvo', params)


# ### Money Flow Index (MFI)
params = {"period": 14, "upper": 80, "lower": 20}
_run_backtest(data_day, 'mfi', params)


# ### Negative Volume Index (NVI)
params = {"sma_period": 25}
_run_backtest(data_week, 'nvi', params)


# ### On-Balance Volume (OBV)
params = {"sma_period": 20}
_run_backtest(data_day, 'obv', params)


# ### Positive Volume Index (PVI)
params = {"sma_period": 25}
_run_backtest(data_week, 'pvi', params)


# ### Percentage Volume Oscillator (PVO)
params = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
_run_backtest(data_day, 'pvo', params)


# ### Volume Flow Indicator (VFI)
params = {"period": 25, "coef": 0.2, "vcoef": 2.5, "smoothing_period": 3}
_run_backtest(data_week, 'vfi', params)


# ### Volume Oscillator (VOO)
params = {"fast_period": 5, "slow_period": 10}
_run_backtest(data_day, 'voo', params)


# ### Volume Price Trend (VPT)
params = {"sma_period": 20}
_run_backtest(data_day, 'vpt', params)


# ### Volume Rate of Change (VRO)
params = {"period": 14}
_run_backtest(data_day, 'vro', params)


# ### Volume Weighted Average Price (VWA)
params = {"window": 20}
_run_backtest(data_day, 'vwa', params)
