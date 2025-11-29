"""Example usage of volume indicators."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import compute_indicator, download_data

# Set pandas display options for better output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ### The Accumulation/Distribution Line (ADL)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='adl',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Accumulation/Distribution Oscillator (ADO)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["period"] = 14
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='ado',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Bill Williams Market Facilitation Index (BWM)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='bwm',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Chaikin Money Flow (CMF)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["period"] = 20
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='cmf',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Ease of Movement (EMV)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["period"] = 14
parameters["divisor"] = 10000
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='emv',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Finite Volume Elements (FVE)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["period"] = 22
parameters["factor"] = 0.3
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='fve',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Force Index (FOI)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["period"] = 13
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='foi',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Klinger Volume Oscillator (KVO)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["fast_period"] = 34
parameters["slow_period"] = 55
parameters["signal_period"] = 13
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='kvo',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Money Flow Index (MFI)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["period"] = 14
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='mfi',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Negative Volume Index (NVI)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["initial_value"] = 1000
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='nvi',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The On-Balance Volume (OBV)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='obv',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Percentage Volume Oscillator (PVO)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["fast_period"] = 12
parameters["slow_period"] = 26
parameters["signal_period"] = 9
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='pvo',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Positive Volume Index (PVI)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["initial_value"] = 1000
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='pvi',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Volume Flow Indicator (VFI)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["period"] = 130
parameters["coef"] = 0.2
parameters["vcoef"] = 2.5
parameters["smoothing_period"] = 3
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='vfi',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Volume Moving Average (VMA)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 14
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='vma',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Volume Oscillator (VOO)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["fast_period"] = 5
parameters["slow_period"] = 10
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='voo',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Volume Price Trend (VPT)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='vpt',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Volume Rate of Change (VRO)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["period"] = 14
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='vro',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Volume Weighted Average Price (VWA)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='vwa',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Williams Accumulation/Distribution (WAD)
# Step 1: Download data
symbol = 'GOOG'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='wad',
    parameters=parameters,
    columns=columns
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)
