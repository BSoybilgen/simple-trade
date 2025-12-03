"""Example usage of volume indicators."""

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


# ### Accumulation/Distribution Line (ADL)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'adl', parameters={}, columns=cols)
_maybe_show(fig)


# ### Accumulation/Distribution Oscillator (ADO)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 14}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'ado', parameters=params, columns=cols)
_maybe_show(fig)


# ### Bill Williams Market Facilitation Index (BWM)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
cols = {"high_col": 'High', "low_col": 'Low', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'bwm', parameters={}, columns=cols)
_maybe_show(fig)


# ### Chaikin Money Flow (CMF)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 20}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'cmf', parameters=params, columns=cols)
_maybe_show(fig)


# ### Ease of Movement (EMV)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 14, "divisor": 10000}
cols = {"high_col": 'High', "low_col": 'Low', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'emv', parameters=params, columns=cols)
_maybe_show(fig)


# ### Finite Volume Elements (FVE)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 22, "factor": 0.3}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'fve', parameters=params, columns=cols)
_maybe_show(fig)


# ### Force Index (FOI)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 13}
cols = {"close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'foi', parameters=params, columns=cols)
_maybe_show(fig)


# ### Klinger Volume Oscillator (KVO)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"fast_period": 34, "slow_period": 55, "signal_period": 13}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'kvo', parameters=params, columns=cols)
_maybe_show(fig)


# ### Money Flow Index (MFI)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 14}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'mfi', parameters=params, columns=cols)
_maybe_show(fig)


# ### Negative Volume Index (NVI)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"initial_value": 1000}
cols = {"close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'nvi', parameters=params, columns=cols)
_maybe_show(fig)


# ### On-Balance Volume (OBV)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
cols = {"close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'obv', parameters={}, columns=cols)
_maybe_show(fig)


# ### Percentage Volume Oscillator (PVO)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
cols = {"volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'pvo', parameters=params, columns=cols)
_maybe_show(fig)


# ### Positive Volume Index (PVI)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"initial_value": 1000}
cols = {"close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'pvi', parameters=params, columns=cols)
_maybe_show(fig)


# ### Volume Flow Indicator (VFI)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {
    "period": 130,
    "coef": 0.2,
    "vcoef": 2.5,
    "smoothing_period": 3,
}
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'vfi', parameters=params, columns=cols)
_maybe_show(fig)


# ### Volume Oscillator (VOO)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"fast_period": 5, "slow_period": 10}
cols = {"volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'voo', parameters=params, columns=cols)
_maybe_show(fig)


# ### Volume Price Trend (VPT)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
cols = {"close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'vpt', parameters={}, columns=cols)
_maybe_show(fig)


# ### Volume Rate of Change (VRO)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
params = {"period": 14}
cols = {"volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'vro', parameters=params, columns=cols)
_maybe_show(fig)


# ### Volume Weighted Average Price (VWA)
data = _fetch_data('GOOG', '2024-01-01', '2025-01-01')
cols = {"high_col": 'High', "low_col": 'Low', "close_col": 'Close', "volume_col": 'Volume'}
_, _, fig = compute_indicator(data, 'vwa', parameters={}, columns=cols)
_maybe_show(fig)
