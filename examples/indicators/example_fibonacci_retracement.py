"""Example usage of Fibonacci retracement levels."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import download_data
from simple_trade import calculate_fibonacci_levels, plot_fibonacci_retracement

# Set pandas display options for better output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ### Fibonacci Retracement Levels
# Step 1: Download data
symbol = 'TSLA'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)
close_data = data['Close']

# Step 2: Calculate Fibonacci retracement levels
fibo = calculate_fibonacci_levels(close_data)
print("\nFibonacci Retracement Levels:")
for level, value in fibo.items():
    print(f"  {level}: {value:.2f}")

# Step 3: Plot Fibonacci retracement
fig = plot_fibonacci_retracement(close_data)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)
