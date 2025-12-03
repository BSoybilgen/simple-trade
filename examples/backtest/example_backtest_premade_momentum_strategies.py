"""Example usage of pre-made momentum indicator backtesting strategies."""

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


# ### Awesome Oscillator (AWO)
params = {"fast_window": 5, "slow_window": 34}
_run_backtest(data_day, 'awo', params)


# ### Balance of Power (BOP)
params = {"window": 14, "smooth": True}
_run_backtest(data_day, 'bop', params)


# ### Commodity Channel Index (CCI)
params = {"window": 20, "constant": 0.015, "upper": 150, "lower": -150}
_run_backtest(data_day, 'cci', params)


# ### Chande Momentum Oscillator (CMO)
params = {"window": 14, "upper": 50, "lower": -50}
_run_backtest(data_day, 'cmo', params)


# ### Center of Gravity (COG)
params = {"window": 20, "signal_window": 9}
_run_backtest(data_day, 'cog', params)


# ### Connors RSI (CRS)
params = {"rsi_window": 3, "streak_window": 2, "rank_window": 100, "upper": 80, "lower": 20}
_run_backtest(data_day, 'crs', params)


# ### Detrended Price Oscillator (DPO)
params = {"window": 20}
_run_backtest(data_day, 'dpo', params)


# ### Elder-Ray Index (ERI)
params = {"window": 25}
_run_backtest(data_day, 'eri', params)


# ### Fisher Transform (FIS)
params = {"window": 25}
_run_backtest(data_day, 'fis', params)


# ### Intraday Momentum Index (IMI)
params = {"window": 14, "upper": 70, "lower": 30}
_run_backtest(data_day, 'imi', params)


# ### Know Sure Thing (KST)
params = {"signal": 9}
_run_backtest(data_day, 'kst', params)


# ### Laguerre RSI (LSI)
params = {"gamma": 0.5, "upper": 80, "lower": 20}
_run_backtest(data_day, 'lsi', params)


# ### Moving Average Convergence Divergence (MACD)
params = {"window_fast": 12, "window_slow": 26, "window_signal": 9}
_run_backtest(data_day, 'mac', params)


# ### Momentum Strength Index (MSI)
params = {"window": 14, "power": 1.0, "upper": 70, "lower": 30}
_run_backtest(data_day, 'msi', params)


# ### Pretty Good Oscillator (PGO)
params = {"window": 14, "upper": 3.0, "lower": -2.0}
_run_backtest(data_day, 'pgo', params)


# ### Percentage Price Oscillator (PPO)
params = {"fast_window": 12, "slow_window": 26, "signal_window": 9}
_run_backtest(data_day, 'ppo', params)


# ### Psychological Line (PSY)
params = {"window": 12, "upper": 70, "lower": 30}
_run_backtest(data_day, 'psy', params)


# ### Qstick (QST)
params = {"window": 10}
_run_backtest(data_day, 'qst', params)


# ### Relative Momentum Index (RMI)
params = {"window": 20, "momentum_period": 5, "upper": 70, "lower": 30}
_run_backtest(data_day, 'rmi', params)


# ### Rate of Change (ROC)
params = {"window": 12}
_run_backtest(data_day, 'roc', params)


# ### Relative Strength Index (RSI)
params = {"window": 14, "upper": 70, "lower": 30}
_run_backtest(data_day, 'rsi', params)


# ### Relative Vigor Index (RVG)
params = {"window": 10}
_run_backtest(data_day, 'rvg', params)


# ### Stochastic RSI (SRI)
params = {"rsi_window": 14, "stoch_window": 14, "k_window": 3, "d_window": 3, "upper": 80, "lower": 20}
_run_backtest(data_day, 'sri', params)


# ### Schaff Trend Cycle (STC)
params = {"window_fast": 23, "window_slow": 50, "cycle": 10, "smooth": 3, "upper": 75, "lower": 25}
_run_backtest(data_day, 'stc', params)


# ### Stochastic Oscillator (STO)
params = {"k_period": 14, "d_period": 3, "smooth_k": 3, "upper": 80, "lower": 20}
_run_backtest(data_day, 'sto', params)


# ### True Strength Index (TSI)
params = {"slow": 25, "fast": 13}
_run_backtest(data_day, 'tsi', params)


# ### TTM Squeeze (TTM)
params = {"length": 20, "std_dev": 2.0, "atr_length": 20, "atr_multiplier": 1.5, "smooth": 3}
_run_backtest(data_day, 'ttm', params)


# ### Ultimate Oscillator (ULT)
params = {"short_window": 7, "medium_window": 14, "long_window": 28, "upper": 65, "lower": 35}
_run_backtest(data_day, 'ult', params)


# ### Vortex Indicator (VOR)
params = {"window": 14}
_run_backtest(data_day, 'vor', params)


# ### Williams Accumulation/Distribution (WAD)
params = {"sma_period": 20}
_run_backtest(data_day, 'wad', params)


# ### Williams %R (WIL)
params = {"window": 14, "upper": -20, "lower": -80}
_run_backtest(data_day, 'wil', params)
