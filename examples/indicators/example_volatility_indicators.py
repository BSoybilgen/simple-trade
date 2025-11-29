"""Example usage of volatility indicators."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import compute_indicator, download_data

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='tkinter')

# Set pandas display options for better output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ### Acceleration Bands (ACB)
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
parameters["factor"] = 0.001
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='acb',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Average True Range (ATR)
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
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='atr',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Average True Range Percent (ATP)
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
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='atp',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Bollinger Band Width (BBW)
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
parameters["window"] = 20
parameters["num_std"] = 2.0
parameters["normalize"] = True
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='bbw',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Bollinger Bands (BOL)
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
parameters["window"] = 20
parameters["num_std"] = 2
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='bol',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Chaikin Volatility (CHA)
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
parameters["ema_window"] = 10
parameters["roc_window"] = 10
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
data, columns, fig = compute_indicator(
    data=data,
    indicator='cha',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Choppiness Index (CHO)
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
data, columns, fig = compute_indicator(
    data=data,
    indicator='cho',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Donchian Channels (DON)
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
parameters["window"] = 20
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
data, columns, fig = compute_indicator(
    data=data,
    indicator='don',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Dynamic Volatility Indicator (DVI)
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
parameters["magnitude_period"] = 5
parameters["stretch_period"] = 100
parameters["smooth_period"] = 3
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='dvi',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Efficiency Ratio (EFR)
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
parameters["period"] = 10
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='efr',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Fractal Dimension Index (FDI)
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
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='fdi',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Garman-Klass Volatility (GRV)
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
parameters["window"] = 20
parameters["annualize"] = True
parameters["trading_periods"] = 252
columns["open_col"] = 'Open'
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='grv',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Heikin-Ashi Volatility (HAV)
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
parameters["method"] = 'std'
columns["open_col"] = 'Open'
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='hav',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Historical Volatility (HIV)
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
parameters["window"] = 20
parameters["annualize"] = True
parameters["trading_periods"] = 252
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='hiv',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Keltner Channels (KEL)
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
parameters["ema_window"] = 20
parameters["atr_window"] = 10
parameters["atr_multiplier"] = 2.0
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='kel',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Median Absolute Deviation (MAD)
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
parameters["scale_factor"] = 1.4826
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='mad',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Mass Index (MAI)
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
parameters["ema_window"] = 9
parameters["sum_window"] = 25
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
data, columns, fig = compute_indicator(
    data=data,
    indicator='mai',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Normalized Average True Range (NAT)
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
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='nat',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Parkinson Volatility (PAV)
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
parameters["window"] = 20
parameters["annualize"] = True
parameters["trading_periods"] = 252
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
data, columns, fig = compute_indicator(
    data=data,
    indicator='pav',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Price Channel Width (PCW)
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
data, columns, fig = compute_indicator(
    data=data,
    indicator='pcw',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Projection Oscillator (PRO)
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
parameters["smooth_period"] = 3
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='pro',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Rogers-Satchell Volatility (RSV)
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
parameters["window"] = 20
parameters["annualize"] = True
parameters["trading_periods"] = 252
columns["open_col"] = 'Open'
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='rsv',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Relative Volatility Index (RVI)
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
parameters["std_window"] = 10
parameters["rsi_window"] = 14
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='rvi',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Standard Deviation (STD)
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
parameters["window"] = 20
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='std',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Stochastic Volatility Indicator (SVI)
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
parameters["atr_period"] = 14
parameters["stoch_period"] = 14
parameters["smooth_k"] = 3
parameters["smooth_d"] = 3
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='svi',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### True Strength Index Volatility (TSI)
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
parameters["long_period"] = 25
parameters["short_period"] = 13
parameters["signal_period"] = 7
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
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

# ### Ulcer Index (ULI)
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
data, columns, fig = compute_indicator(
    data=data,
    indicator='uli',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Vertical Horizontal Filter (VHF)
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
parameters["period"] = 28
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='vhf',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Volatility Quality Index (VQI)
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
parameters["window"] = 9
parameters["smooth_window"] = 9
columns["open_col"] = 'Open'
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
columns["volume_col"] = 'Volume'
data, columns, fig = compute_indicator(
    data=data,
    indicator='vqi',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Volatility Ratio (VOR)
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
parameters["short_period"] = 6
parameters["long_period"] = 100
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

# ### Volatility Switch Index (VSI)
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
parameters["short_period"] = 10
parameters["long_period"] = 50
parameters["threshold"] = 1.2
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='vsi',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)
