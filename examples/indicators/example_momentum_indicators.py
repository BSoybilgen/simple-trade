"""Example usage of momentum indicators originally provided in the notebook version."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import compute_indicator, download_data

# Set pandas display options for better output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ### The Commodity Channel Index (CCI)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 20
parameters["constant"] = 0.015
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='cci',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Chande Momentum Oscillator (CMO)
# Step 1: Download data
symbol = 'MSFT'
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
data, columns, fig = compute_indicator(
    data=data,
    indicator='cmo',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Center of Gravity (COG)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 10
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='cog',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Connors RSI (CRS)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["rsi_window"] = 3
parameters["streak_window"] = 2
parameters["rank_window"] = 100
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='crs',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Detrended Price Oscillator (DPO)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 20
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='dpo',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Elder-Ray Index (ERI)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 13
columns["close_col"] = 'Close'
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
data, columns, fig = compute_indicator(
    data=data,
    indicator='eri',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Fisher Transform (FIS)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 9
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
data, columns, fig = compute_indicator(
    data=data,
    indicator='fis',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Know Sure Thing (KST)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["roc_periods"] = (10, 15, 20, 30)
parameters["ma_periods"] = (10, 10, 10, 15)
parameters["weights"] = (1, 2, 3, 4)
parameters["signal"] = 9
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='kst',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Laguerre RSI (LSI)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["gamma"] = 0.5
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='lsi',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Moving Average Convergence Divergence Index (MACD), Signal Line, and Histogram
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window_fast"] = 12
parameters["window_slow"] = 26
parameters["window_signal"] = 9
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='mac',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Momentum Strength Index (MSI)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 14
parameters["power"] = 1.0
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='msi',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Qstick Indicator (QST)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 10
columns["close_col"] = 'Close'
columns["open_col"] = 'Open'
data, columns, fig = compute_indicator(
    data=data,
    indicator='qst',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Relative Momentum Index (RMI)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 20
parameters["momentum_period"] = 5
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='rmi',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Rate of Change (ROC)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 12
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='roc',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Relative Strength Index (RSI)
# Step 1: Download data
symbol = 'MSFT'
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
data, columns, fig = compute_indicator(
    data=data,
    indicator='rsi',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Schaff Trend Cycle (STC)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window_fast"] = 23
parameters["window_slow"] = 50
parameters["cycle"] = 10
parameters["smooth"] = 3
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='stc',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Stochastic Oscillator Index (STOCH)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["k_period"] = 14
parameters["d_period"] = 3
parameters["smooth_k"] = 3
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='sto',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The True Strength Index (TSI)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["slow"] = 25
parameters["fast"] = 13
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='tsi',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The TTM Squeeze Momentum Indicator (TTM)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["length"] = 20
parameters["std_dev"] = 2.0
parameters["atr_length"] = 20
parameters["atr_multiplier"] = 1.5
parameters["smooth"] = 3
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='ttm',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Ultimate Oscillator (ULT)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["short_window"] = 7
parameters["medium_window"] = 14
parameters["long_window"] = 28
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='ult',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Vortex Indicator (VOR)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 14
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='vor',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Williams %R (WIL)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 14
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='wil',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)
