"""Example usage of trend indicators."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import IndicatorPlotter, compute_indicator, download_data

# Set pandas display options for better output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

plotter = IndicatorPlotter()

# ### The Average Directional Index (ADX)
# Step 1: Download data
symbol = 'TSLA'
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
    indicator='adx',
    figure=False,
    parameters=parameters,
    columns=columns,
)

window = parameters["window"]
fig = plotter.plot_results(
    data,
    price_col='Close',
    column_names=columns,
    plot_on_subplot=True,
    title=f"{symbol} with ADX({window})",
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Adaptive Deviation-Scaled Moving Average (ADSMA)
# Step 1: Download data
symbol = 'AAPL'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 40
parameters["sensitivity"] = 50.0
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='ads',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Aroon indicator (AROON)
# Step 1: Download data
symbol = 'ETH-USD'
start = '2023-01-01'
end = '2025-01-01'
interval = '1wk'
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
    indicator='aro',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Adaptive Moving Average (AMA)
# Step 1: Download data
symbol = 'AAPL'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 10
parameters["fast_period"] = 2
parameters["slow_period"] = 30
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='ama',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Arnaud Legoux Moving Average (ALMA)
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
parameters["sigma"] = 6
parameters["offset"] = 0.85
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='alm',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Exponential Moving Average (EMA)
# Step 1: Download data
symbol = 'ETH-USD'
start = '2023-01-01'
end = '2025-01-01'
interval = '1wk'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 20
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='ema',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Ehlers Instantaneous Trendline (EIT)
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
parameters["alpha"] = 0.07
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='eit',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Double Exponential Moving Average (DEMA)
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
    indicator='dem',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Ehlers Adaptive CyberCycle (EAC)
# Step 1: Download data
symbol = 'TSLA'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["alpha"] = 0.07
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='eac',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Fractal Adaptive Moving Average (FMA)
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
parameters["window"] = 16
parameters["alpha_floor"] = 0.01
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='fma',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Guppy Multiple Moving Average (GMA)
# Step 1: Download data
symbol = 'TSLA'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["short_windows"] = (3, 5, 8, 10, 12, 15)
parameters["long_windows"] = (30, 35, 40, 45, 50, 60)
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='gma',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Hull Moving Average (HMA)
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
parameters["window"] = 21
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='hma',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Hilbert Transform Trendline (HTT)
# Step 1: Download data
symbol = 'AAPL'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 16
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='htt',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Ichimoku Cloud indicators (ICHI)
# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-06-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["tenkan_period"] = 9
parameters["kijun_period"] = 26
parameters["senkou_b_period"] = 52
parameters["displacement"] = 26
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='ich',
    figure=False,
    plot_type='candlestick',
    parameters=parameters,
    columns=columns,
)

tenkan_period = parameters["tenkan_period"]
kijun_period = parameters["kijun_period"]
senkou_b_period = parameters["senkou_b_period"]
displacement = parameters["displacement"]
fig = plotter.plot_results(
    data,
    price_col='Close',
    column_names=columns,
    plot_on_subplot=False,
    plot_type='candlestick',
    title=(
        f"{symbol} with Ichimoku Cloud"
        f" ({tenkan_period}, {kijun_period}, {senkou_b_period}, {displacement})"
    ),
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Jurík Moving Average (JMA)
# Step 1: Download data
symbol = 'ETH-USD'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["length"] = 21
parameters["phase"] = 0.0
parameters["power"] = 2.0
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='jma',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Kaufman Adaptive Moving Average (KMA)
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
parameters["window"] = 10
parameters["fast_period"] = 2
parameters["slow_period"] = 30
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='kma',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Least Squares Moving Average (LSMA)
# Step 1: Download data
symbol = 'ETH-USD'
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
    indicator='lsm',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The McGinley Dynamic (MGD)
# Step 1: Download data
symbol = 'AAPL'
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
    indicator='mgd',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Parabolic SAR (PSAR)
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
parameters["af_initial"] = 0.02
parameters["af_step"] = 0.02
parameters["af_max"] = 0.2
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='psa',
    figure=False,
    parameters=parameters,
    columns=columns,
)

af_initial = parameters["af_initial"]
af_step = parameters["af_step"]
af_max = parameters["af_max"]
psar_columns = [f'PSAR_{af_initial}_{af_step}_{af_max}']
fig = plotter.plot_results(
    data,
    price_col='Close',
    column_names=psar_columns,
    plot_on_subplot=False,
    title=f"{symbol} with PSAR({af_initial}, {af_step}, {af_max})",
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Simple Moving Average (SMA)
# Step 1: Download data
symbol = 'AAPL'
start = '2024-01-01'
end = '2024-04-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 50
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='sma',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Smoothed Moving Average (SOA)
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
    indicator='soa',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Sine Weighted Moving Average (SWMA)
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
    indicator='swm',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The SuperTrend (STREND)
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
parameters["period"] = 7
parameters["multiplier"] = 3.0
columns["high_col"] = 'High'
columns["low_col"] = 'Low'
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='str',
    figure=True,
    plot_type='line',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Triple Exponential Average (TRIX)
# Step 1: Download data
symbol = 'AAPL'
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
    indicator='tri',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Triangular Moving Average (TMA)
# Step 1: Download data
symbol = 'ETH-USD'
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
    indicator='tma',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Triple Exponential Moving Average (TEMA)
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
    indicator='tem',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Variable Index Dynamic Average (VID)
# Step 1: Download data
symbol = 'AAPL'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 21
parameters["cmo_window"] = 9
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='vid',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Weighted Moving Average (WMA)
# Step 1: Download data
symbol = 'AAPL'
start = '2024-01-01'
end = '2025-01-01'
interval = '1wk'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
parameters = dict()
columns = dict()
parameters["window"] = 20
columns["close_col"] = 'Close'
data, columns, fig = compute_indicator(
    data=data,
    indicator='wma',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### The Zero-Lag Moving Average (ZMA)
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
    indicator='zma',
    parameters=parameters,
    columns=columns,
)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)
