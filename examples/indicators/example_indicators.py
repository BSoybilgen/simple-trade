import sys
import os

import simple_trade
from simple_trade import compute_indicator, download_data
import pandas as pd
from simple_trade import IndicatorPlotter

# Set pandas display options for better output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Create a plotter instance
plotter = IndicatorPlotter()

### The ICHI CLOUD

# Step 1: Download data
symbol = 'MSFT'
start = '2024-01-01'
end = '2025-06-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)

# Step 2: Calculate indicator
data = compute_indicator(
    data=data,
    indicator='ichimoku',
    tenkan_period=9, 
    kijun_period=26,
    senkou_b_period=52, 
    displacement=26,
    high_col='High',
    low_col='Low',
    close_col='Close'
)