"""Example usage of trend indicators."""

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


# ### Average Directional Index (ADX)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'adx', parameters=params, columns=cols)
_maybe_show(fig)


# ### Aroon (ARO)
data = _fetch_data('AAPL', '2024-01-01', '2025-01-01')
params = {"period": 25}
cols = {"high_col": 'High', "low_col": 'Low'}
_, _, fig = compute_indicator(data, 'aro', parameters=params, columns=cols)
_maybe_show(fig)


# ### Ehlers Adaptive CyberCycle (EAC)
data = _fetch_data('TSLA', '2024-01-01', '2025-01-01')
params = {"alpha": 0.07}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'eac', parameters=params, columns=cols)
_maybe_show(fig)


# ### Ehlers Instantaneous Trendline (EIT)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"alpha": 0.07}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'eit', parameters=params, columns=cols)
_maybe_show(fig)


# ### Hilbert Transform Trendline (HTT)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 16}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'htt', parameters=params, columns=cols)
_maybe_show(fig)


# ### Ichimoku Cloud (ICH)
data = _fetch_data('MSFT', '2024-01-01', '2025-06-01')
params = {
    "tenkan_period": 9,
    "kijun_period": 26,
    "senkou_b_period": 52,
    "displacement": 26,
}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'ich', parameters=params, columns=cols)
_maybe_show(fig)


# ### McGinley Dynamic (MGD)
data = _fetch_data('AAPL', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'mgd', parameters=params, columns=cols)
_maybe_show(fig)


# ### Projection Oscillator (PRO)
data = _fetch_data('TSLA', '2024-01-01', '2025-01-01')
params = {"period": 10, "smooth_period": 3}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'pro', parameters=params, columns=cols)
_maybe_show(fig)


# ### Parabolic SAR (PSA)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"af_initial": 0.02, "af_step": 0.02, "af_max": 0.2}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'psa', parameters=params, columns=cols)
_maybe_show(fig)


# ### SuperTrend (STR)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 7, "multiplier": 3.0}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'str', parameters=params, columns=cols)
_maybe_show(fig)


# ### Triple Exponential Average (TRI)
data = _fetch_data('AAPL', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'tri', parameters=params, columns=cols)
_maybe_show(fig)


# ### Volatility Quality Index (VQI)
data = _fetch_data('TSLA', '2024-01-01', '2025-01-01')
params = {"period": 9, "smooth_period": 9, "volatility_cutoff": 0.1}
cols = {
    "high_col": 'High',
    "low_col": 'Low',
    "close_col": 'Close',
    "volume_col": 'Volume',
}
_, _, fig = compute_indicator(data, 'vqi', parameters=params, columns=cols)
_maybe_show(fig)
