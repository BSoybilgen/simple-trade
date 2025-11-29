import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_pivot_points(data: pd.Series, window: int = 10) -> pd.Series:
    """Find pivot points (highs and lows) in a time series using a vectorized approach.

    Args:
        data (pd.Series): Time series data (e.g., close prices).
        window (int): The number of periods to look on each side to determine a pivot.

    Returns:
        pd.Series: A series with pivot points, where 1 indicates a pivot high,
                   -1 indicates a pivot low, and 0 indicates no pivot.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input 'data' must be a pandas Series.")

    # The total window size for the rolling operation
    rolling_window_size = 2 * window + 1

    # Find local maxima: a point is a local max if it's the maximum in its window
    local_max = data.rolling(window=rolling_window_size, center=True, min_periods=1).max()
    is_pivot_high = (data == local_max)

    # Find local minima: a point is a local min if it's the minimum in its window
    local_min = data.rolling(window=rolling_window_size, center=True, min_periods=1).min()
    is_pivot_low = (data == local_min)

    # Initialize pivots series
    pivots = pd.Series(0, index=data.index, dtype=int)

    # Set pivot highs and lows
    pivots[is_pivot_high] = 1
    pivots[is_pivot_low] = -1

    return pivots


def find_resistance_support_lines(data: pd.Series, window: int = 10, tolerance: float = 0.02):
    """
    Finds resistance and support lines from pivot points.

    Args:
        data (pd.Series): The price data.
        window (int): The number of periods to look on each side to determine a pivot.
        tolerance (float): The percentage tolerance to group pivot points into a line.

    Returns:
        tuple: A tuple containing two lists: resistance_lines and support_lines.
    """
    pivots = find_pivot_points(data, window)
    pivot_highs = data[pivots == 1]
    pivot_lows = data[pivots == -1]

    resistance_lines = []
    if not pivot_highs.empty:
        sorted_highs = sorted(pivot_highs)
        current_line = sorted_highs[0]
        line_points = [current_line]
        for price in sorted_highs[1:]:
            if price <= current_line * (1 + tolerance):
                line_points.append(price)
                current_line = np.mean(line_points)
            else:
                resistance_lines.append(current_line)
                current_line = price
                line_points = [current_line]
        resistance_lines.append(current_line)

    support_lines = []
    if not pivot_lows.empty:
        sorted_lows = sorted(pivot_lows)
        current_line = sorted_lows[0]
        line_points = [current_line]
        for price in sorted_lows[1:]:
            if price <= current_line * (1 + tolerance):
                line_points.append(price)
                current_line = np.mean(line_points)
            else:
                support_lines.append(current_line)
                current_line = price
                line_points = [current_line]
        support_lines.append(current_line)

    return resistance_lines, support_lines, pivots


def plot_resistance_support(data: pd.Series, window: int = 10, tolerance: float = 0.02):
    """
    Plots the price data, pivot points, and resistance/support lines.

    Args:
        data (pd.Series): The price data.
        window (int): The number of periods to look on each side to determine a pivot.
        tolerance (float): The percentage tolerance to group pivot points into a line.
    """
    resistance_lines, support_lines, pivots = find_resistance_support_lines(data, window, tolerance)
    plt.figure(figsize=(15, 8))
    plt.plot(data, label='Price')

    pivot_highs = data[pivots == 1]
    pivot_lows = data[pivots == -1]

    plt.scatter(pivot_highs.index, pivot_highs, color='r', marker='^', label='Pivot Highs')
    plt.scatter(pivot_lows.index, pivot_lows, color='g', marker='v', label='Pivot Lows')

    for line in resistance_lines:
        plt.axhline(line, color='r', linestyle='--', alpha=0.5)

    for line in support_lines:
        plt.axhline(line, color='g', linestyle='--', alpha=0.5)

    plt.title('Resistance and Support Lines')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
