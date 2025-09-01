import pandas as pd
import matplotlib.pyplot as plt

def calculate_fibonacci_levels(data: pd.Series):
    """
    Calculates Fibonacci retracement levels.

    Args:
        data (pd.Series): A pandas Series of price data.

    Returns:
        dict: A dictionary containing the Fibonacci levels.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input 'data' must be a pandas Series.")

    high = data.max()
    low = data.min()
    price_range = high - low

    levels = {
        'level_0': high,
        'level_236': high - 0.236 * price_range,
        'level_382': high - 0.382 * price_range,
        'level_500': high - 0.5 * price_range,
        'level_618': high - 0.618 * price_range,
        'level_786': high - 0.786 * price_range,
        'level_100': low,
    }
    return levels

def plot_fibonacci_retracement(data: pd.Series):
    """
    Plots the price data along with Fibonacci retracement levels.

    Args:
        data (pd.Series): The original price data.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data, label='Price')

    colors = {
        'level_0': 'red',
        'level_236': 'orange',
        'level_382': 'yellow',
        'level_500': 'green',
        'level_618': 'blue',
        'level_786': 'indigo',
        'level_100': 'violet'
    }

    levels = calculate_fibonacci_levels(data)

    for level_name, level_price in levels.items():
        plt.axhline(y=level_price, color=colors.get(level_name, 'gray'), linestyle='--', label=f'{level_name.replace("_", " ").title()} ({level_price:.2f})')

    plt.title('Fibonacci Retracement')
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.show()
