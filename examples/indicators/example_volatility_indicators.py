"""Example usage of volatility indicators."""

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


# ### Acceleration Bands (ACB)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 20, "factor": 0.001}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'acb', parameters=params, columns=cols)
_maybe_show(fig)

# ### Average True Range (ATR)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'atr', parameters=params, columns=cols)
_maybe_show(fig)

# ### Average True Range Percent (ATP)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'atp', parameters=params, columns=cols)
_maybe_show(fig)

# ### Bollinger Band Width (BBW)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 20, "num_std": 2.0, "normalize": True}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'bbw', parameters=params, columns=cols)
_maybe_show(fig)

# ### Bollinger Bands (BOL)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 20, "num_std": 2}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'bol', parameters=params, columns=cols)
_maybe_show(fig)

# ### Chaikin Volatility (CHA)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"ema_window": 10, "roc_window": 10}
cols = {"high_col": 'High', "low_col": 'Low'}
_, _, fig = compute_indicator(data, 'cha', parameters=params, columns=cols)
_maybe_show(fig)

# ### Choppiness Index (CHO)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 14}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'cho', parameters=params, columns=cols)
_maybe_show(fig)

# ### Donchian Channels (DON)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 20}
cols = {"high_col": 'High', "low_col": 'Low'}
_, _, fig = compute_indicator(data, 'don', parameters=params, columns=cols)
_maybe_show(fig)

# ### Dynamic Volatility Indicator (DVI)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"magnitude_period": 5, "stretch_period": 100, "smooth_period": 3}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'dvi', parameters=params, columns=cols)
_maybe_show(fig)

# ### Efficiency Ratio (EFR)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 10}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'efr', parameters=params, columns=cols)
_maybe_show(fig)

# ### Fractal Dimension Index (FDI)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'fdi', parameters=params, columns=cols)
_maybe_show(fig)

# ### Garman-Klass Volatility (GRV)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 20, "annualize": True, "trading_periods": 252}
cols = {
    "open_col": 'Open',
    "high_col": 'High',
    "low_col": 'Low',
    "close_col": 'Close',
}
_, _, fig = compute_indicator(data, 'grv', parameters=params, columns=cols)
_maybe_show(fig)

# ### Heikin-Ashi Volatility (HAV)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 14, "method": 'std'}
cols = {
    "open_col": 'Open',
    "high_col": 'High',
    "low_col": 'Low',
    "close_col": 'Close',
}
_, _, fig = compute_indicator(data, 'hav', parameters=params, columns=cols)
_maybe_show(fig)

# ### Historical Volatility (HIV)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 20, "annualize": True, "trading_periods": 252}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'hiv', parameters=params, columns=cols)
_maybe_show(fig)

# ### Keltner Channels (KEL)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"ema_window": 20, "atr_window": 10, "atr_multiplier": 2.0}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'kel', parameters=params, columns=cols)
_maybe_show(fig)

# ### Median Absolute Deviation (MAD)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 20, "scale_factor": 1.4826}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'mad', parameters=params, columns=cols)
_maybe_show(fig)

# ### Mass Index (MAI)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"ema_window": 9, "sum_window": 25}
cols = {"high_col": 'High', "low_col": 'Low'}
_, _, fig = compute_indicator(data, 'mai', parameters=params, columns=cols)
_maybe_show(fig)

# ### Normalized Average True Range (NAT)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'nat', parameters=params, columns=cols)
_maybe_show(fig)

# ### Parkinson Volatility (PAV)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 20, "annualize": True, "trading_periods": 252}
cols = {"high_col": 'High', "low_col": 'Low'}
_, _, fig = compute_indicator(data, 'pav', parameters=params, columns=cols)
_maybe_show(fig)

# ### Price Channel Width (PCW)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 20}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'pcw', parameters=params, columns=cols)
_maybe_show(fig)

# ### Rogers-Satchell Volatility (RSV)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 20, "annualize": True, "trading_periods": 252}
cols = {
    "open_col": 'Open',
    "high_col": 'High',
    "low_col": 'Low',
    "close_col": 'Close',
}
_, _, fig = compute_indicator(data, 'rsv', parameters=params, columns=cols)
_maybe_show(fig)

# ### Relative Volatility Index (RVI)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"std_window": 10, "rsi_window": 14}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'rvi', parameters=params, columns=cols)
_maybe_show(fig)

# ### Stochastic Volatility Indicator (SVI)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {
    "atr_period": 14,
    "stoch_period": 14,
    "smooth_k": 3,
    "smooth_d": 3,
}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'svi', parameters=params, columns=cols)
_maybe_show(fig)

# ### True Strength Index Volatility (TSV)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"long_period": 25, "short_period": 13, "atr_period": 14}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close'}
_, _, fig = compute_indicator(data, 'tsv', parameters=params, columns=cols)
_maybe_show(fig)

# ### Ulcer Index (ULI)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"window": 14}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'uli', parameters=params, columns=cols)
_maybe_show(fig)

# ### Vertical Horizontal Filter (VHF)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 28}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'vhf', parameters=params, columns=cols)
_maybe_show(fig)

# ### Volatility Ratio (VRA)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"short_period": 5, "long_period": 20}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'vra', parameters=params, columns=cols)
_maybe_show(fig)


# ### Volatility Switch Index (VSI)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"short_period": 10, "long_period": 50, "threshold": 1.2}
cols = {"close_col": 'Close'}
_, _, fig = compute_indicator(data, 'vsi', parameters=params, columns=cols)
_maybe_show(fig)
