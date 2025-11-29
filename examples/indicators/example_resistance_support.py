"""Example usage of resistance and support lines."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import download_data
from simple_trade import find_resistance_support_lines, plot_resistance_support

# Set pandas display options for better output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ### Resistance and Support Lines
# Step 1: Download data
symbol = 'TSLA'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)
close_data = data['Close']

# Step 2: Find resistance and support lines
resistance_lines, support_lines, pivots = find_resistance_support_lines(close_data)
print(f"\nFound {len(resistance_lines)} resistance lines and {len(support_lines)} support lines")

# Step 3: Plot resistance and support lines
fig = plot_resistance_support(close_data)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)
