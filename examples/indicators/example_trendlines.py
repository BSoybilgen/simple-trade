"""Example usage of automatic trendlines."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import download_data
from simple_trade import find_best_trendlines, plot_trendlines

# Set pandas display options for better output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ### Automatic Trendlines
# Step 1: Download data
symbol = 'TSLA'
start = '2024-01-01'
end = '2025-01-01'
interval = '1d'
print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start, end, interval=interval)
close_data = data['Close']

# Step 2: Find trendlines
result_df, uptrend_lines, downtrend_lines = find_best_trendlines(close_data, window=10, min_touches=3, tolerance=0.02)
print(f"\nFound {len(uptrend_lines)} uptrend lines and {len(downtrend_lines)} downtrend lines")

# Display the DataFrame with Close and Trendline columns
print("\nTrendline Data (first 10 rows):")
print(result_df.head(10))

# Print details of uptrend lines
if uptrend_lines:
    print("\nUptrend Lines:")
    for i, line in enumerate(uptrend_lines):
        print(f"  Line {i+1}: {line['touches']} touches, slope={line['slope']:.4f}")

# Print details of downtrend lines
if downtrend_lines:
    print("\nDowntrend Lines:")
    for i, line in enumerate(downtrend_lines):
        print(f"  Line {i+1}: {line['touches']} touches, slope={line['slope']:.4f}")

# Step 3: Plot trendlines
fig = plot_trendlines(close_data, window=10, min_touches=3, tolerance=0.02)
if fig is not None:
    plt.show(block=True)
    plt.close(fig)

print(result_df.head())