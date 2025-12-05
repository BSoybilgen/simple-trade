"""Example usage of moving-average indicators."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import compute_indicator, download_data

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


# ### Adaptive Deviation-Scaled Moving Average (ADS)
data = _fetch_data('AAPL', '2024-01-01', '2025-01-01')
params = {"window": 20, "sensitivity": 0.5}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'ads', parameters=params, columns=cols)
_maybe_show(fig)


# ### Arnaud Legoux Moving Average (ALM)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 20, "sigma": 6.0, "offset": 0.85}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'alm', parameters=params, columns=cols)
_maybe_show(fig)


# ### Adaptive Moving Average (AMA)
data = _fetch_data('AAPL', '2024-01-01', '2025-01-01')
params = {"window": 10, "fast_sc": 2, "slow_sc": 30}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'ama', parameters=params, columns=cols)
_maybe_show(fig)


# ### Double Exponential Moving Average (DEM)
data = _fetch_data('TSLA', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'dem', parameters=params, columns=cols)
_maybe_show(fig)


# ### Exponential Moving Average (EMA)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'ema', parameters=params, columns=cols)
_maybe_show(fig)


# ### Elastic Volume Weighted Moving Average (EVW)
data = _fetch_data('AAPL', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'evw', parameters=params, columns=cols)
_maybe_show(fig)


# ### Fractal Adaptive Moving Average (FMA)
data = _fetch_data('AAPL', '2024-01-01', '2025-01-01')
params = {"window": 16, "alpha_floor": 0.01}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'fma', parameters=params, columns=cols)
_maybe_show(fig)


# ### Guppy Multiple Moving Average (GMA)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"short_windows": (3, 5, 8, 10, 12, 15), "long_windows": (30, 35, 40, 45, 50, 60)}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'gma', parameters=params, columns=cols)
_maybe_show(fig)


# ### Hull Moving Average (HMA)
data = _fetch_data('TSLA', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'hma', parameters=params, columns=cols)
_maybe_show(fig)


# ### Jurik Moving Average (JMA)
data = _fetch_data('AAPL', '2024-01-01', '2025-01-01')
params = {"length": 21, "phase": 0, "power": 2.0}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'jma', parameters=params, columns=cols)
_maybe_show(fig)


# ### Least Squares Moving Average (LSM)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 25}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'lsm', parameters=params, columns=cols)
_maybe_show(fig)


# ### MESA Adaptive Moving Average (MAM)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"fast_limit": 0.5, "slow_limit": 0.05}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'mam', parameters=params, columns=cols)
_maybe_show(fig)


# ### Simple Moving Average (SMA)
data = _fetch_data('TSLA', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'sma', parameters=params, columns=cols)
_maybe_show(fig)


# ### Smoothed Moving Average (SOA)
data = _fetch_data('AAPL', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'soa', parameters=params, columns=cols)
_maybe_show(fig)


# ### Sine-Weighted Moving Average (SWM)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'swm', parameters=params, columns=cols)
_maybe_show(fig)


# ### Triple Exponential Moving Average (TEM)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'tem', parameters=params, columns=cols)
_maybe_show(fig)


# ### Triangular Moving Average (TMA)
data = _fetch_data('AAPL', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'tma', parameters=params, columns=cols)
_maybe_show(fig)


# ### Time Series Forecast (TSF)
data = _fetch_data('TSLA', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'tsf', parameters=params, columns=cols)
_maybe_show(fig)


# ### T3 Moving Average (TT3)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 5, "v_factor": 0.7}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'tt3', parameters=params, columns=cols)
_maybe_show(fig)


# ### Variable Index Dynamic Average (VID)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 21, "cmo_window": 9}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'vid', parameters=params, columns=cols)
_maybe_show(fig)


# ### Volume Moving Average (VMA)
data = _fetch_data('TSLA', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'vma', parameters=params, columns=cols)
_maybe_show(fig)


# ### Weighted Moving Average (WMA)
data = _fetch_data('AAPL', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'wma', parameters=params, columns=cols)
_maybe_show(fig)


# ### Zero-Lag Moving Average (ZMA)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'zma', parameters=params, columns=cols)
_maybe_show(fig)