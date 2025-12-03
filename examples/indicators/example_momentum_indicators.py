"""Example usage of momentum indicators."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import compute_indicator, download_data

# Configure pandas display to make console output easier to scan
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def _fetch_data(symbol: str, start: str, end: str, interval: str = '1d'):
    print(f"\nDownloading data for {symbol} ({interval})...")
    return download_data(symbol, start, end, interval=interval)


def _maybe_show(fig):
    if fig is not None:
        plt.show(block=True)
        plt.close(fig)


# ### Awesome Oscillator (AWO)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"fast_window": 5, "slow_window": 34}
cols = {"high_col": 'High', "low_col": 'Low'}
_, _, fig = compute_indicator(data, 'awo', parameters=params, columns=cols)
_maybe_show(fig)


# ### Balance of Power (BOP)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 14, "smooth": True}
cols = {"open_col": 'Open', "high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'bop', parameters=params, columns=cols)
_maybe_show(fig)


# ### Commodity Channel Index (CCI)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 20, "constant": 0.015}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'cci', parameters=params, columns=cols)
_maybe_show(fig)


# ### Chande Momentum Oscillator (CMO)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'cmo', parameters=params, columns=cols)
_maybe_show(fig)


# ### Center of Gravity (COG)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 10}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'cog', parameters=params, columns=cols)
_maybe_show(fig)


# ### Connors RSI (CRS)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"rsi_window": 3, "streak_window": 2, "rank_window": 100}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'crs', parameters=params, columns=cols)
_maybe_show(fig)


# ### Detrended Price Oscillator (DPO)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'dpo', parameters=params, columns=cols)
_maybe_show(fig)


# ### Elder-Ray Index (ERI)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 13}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'eri', parameters=params, columns=cols)
_maybe_show(fig)


# ### Fisher Transform (FIS)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 9}
cols = {"high_col": 'High', "low_col": 'Low'}
_, _, fig = compute_indicator(data, 'fis', parameters=params, columns=cols)
_maybe_show(fig)


# ### Intraday Momentum Index (IMI)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"open_col": 'Open', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'imi', parameters=params, columns=cols)
_maybe_show(fig)


# ### Know Sure Thing (KST)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {
    "roc_periods": (10, 15, 20, 30),
    "ma_periods": (10, 10, 10, 15),
    "weights": (1, 2, 3, 4),
    "signal": 9,
}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'kst', parameters=params, columns=cols)
_maybe_show(fig)


# ### Laguerre RSI (LSI)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"gamma": 0.5}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'lsi', parameters=params, columns=cols)
_maybe_show(fig)


# ### Moving Average Convergence Divergence (MACD)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window_fast": 12, "window_slow": 26, "window_signal": 9}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'mac', parameters=params, columns=cols)
_maybe_show(fig)


# ### Momentum Strength Index (MSI)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 14, "power": 1.0}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'msi', parameters=params, columns=cols)
_maybe_show(fig)


# ### Pretty Good Oscillator (PGO)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'pgo', parameters=params, columns=cols)
_maybe_show(fig)


# ### Percentage Price Oscillator (PPO)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"fast_window": 12, "slow_window": 26, "signal_window": 9}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'ppo', parameters=params, columns=cols)
_maybe_show(fig)


# ### Psychological Line (PSY)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 12}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'psy', parameters=params, columns=cols)
_maybe_show(fig)


# ### Qstick Indicator (QST)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 10}
cols = {"open_col": 'Open', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'qst', parameters=params, columns=cols)
_maybe_show(fig)


# ### Relative Momentum Index (RMI)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 20, "momentum_period": 5}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'rmi', parameters=params, columns=cols)
_maybe_show(fig)


# ### Rate of Change (ROC)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 12}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'roc', parameters=params, columns=cols)
_maybe_show(fig)


# ### Relative Strength Index (RSI)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'rsi', parameters=params, columns=cols)
_maybe_show(fig)


# ### Relative Vigor Index (RVG)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 10}
cols = {"open_col": 'Open', "high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'rvg', parameters=params, columns=cols)
_maybe_show(fig)


# ### Stochastic RSI (SRI)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"rsi_window": 14, "stoch_window": 14, "k_window": 3, "d_window": 3}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'sri', parameters=params, columns=cols)
_maybe_show(fig)


# ### Schaff Trend Cycle (STC)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window_fast": 23, "window_slow": 50, "cycle": 10, "smooth": 3}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'stc', parameters=params, columns=cols)
_maybe_show(fig)


# ### Stochastic Oscillator (STO)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"k_period": 14, "d_period": 3, "smooth_k": 3}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'sto', parameters=params, columns=cols)
_maybe_show(fig)


# ### True Strength Index (TSI)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"slow": 25, "fast": 13}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'tsi', parameters=params, columns=cols)
_maybe_show(fig)


# ### TTM Squeeze Momentum Indicator (TTM)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {
    "length": 20,
    "std_dev": 2.0,
    "atr_length": 20,
    "atr_multiplier": 1.5,
    "smooth": 3,
}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'ttm', parameters=params, columns=cols)
_maybe_show(fig)


# ### Ultimate Oscillator (ULT)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"short_window": 7, "medium_window": 14, "long_window": 28}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'ult', parameters=params, columns=cols)
_maybe_show(fig)


# ### Vortex Indicator (VOR)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'vor', parameters=params, columns=cols)
_maybe_show(fig)


# ### Williams Accumulation/Distribution (WAD)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"sma_period": 20}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'wad', parameters=params, columns=cols)
_maybe_show(fig)


# ### Williams %R (WIL)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'wil', parameters=params, columns=cols)
_maybe_show(fig)
