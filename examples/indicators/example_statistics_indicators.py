"""Example usage of statistics indicators."""

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


# ### Kurtosis (KUR)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'kur', parameters=params, columns=cols)
_maybe_show(fig)


# ### Mean Absolute Deviation (MAB)
data = _fetch_data('TSLA', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'mab', parameters=params, columns=cols)
_maybe_show(fig)


# ### Median (MED)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'med', parameters=params, columns=cols)
_maybe_show(fig)


# ### Quartiles (QUA)
data = _fetch_data('AAPL', '2024-01-01', '2025-01-01')
params = {"window": 20, "lower_pct": 25, "upper_pct": 75}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'qua', parameters=params, columns=cols)
_maybe_show(fig)


# ### Skewness (SKW)
data = _fetch_data('MSFT', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'skw', parameters=params, columns=cols)
_maybe_show(fig)


# ### Standard Deviation (STD)
data = _fetch_data('AAPL', '2024-01-01', '2025-01-01')
params = {"window": 20, "ddof": 0}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'std', parameters=params, columns=cols)
_maybe_show(fig)


# ### Variance (VAR)
data = _fetch_data('TSLA', '2024-01-01', '2025-01-01')
params = {"window": 20, "ddof": 0}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'var', parameters=params, columns=cols)
_maybe_show(fig)


# ### Z-Score (ZSC)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'zsc', parameters=params, columns=cols)
_maybe_show(fig)
