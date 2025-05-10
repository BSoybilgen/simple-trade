import pandas as pd
import numpy as np


def aroon(df: pd.DataFrame, period: int = 14, high_col: str = 'High', low_col: str = 'Low') -> tuple:
    """
    Calculates the Aroon indicator, which measures the time it takes for a security
    to reach its highest and lowest points over a specified time period.
    
    Args:
        df (pd.DataFrame): The dataframe containing price data. Must have high and low columns.
        period (int): The lookback period for calculation. Default is 14.
        high_col (str): The name of the high price column (default: 'High').
        low_col (str): The name of the low price column (default: 'Low').
    
    Returns:
        tuple: A tuple containing (aroon_up, aroon_down, aroon_oscillator).
    
    The Aroon indicator consists of three components:
    
    1. Aroon Up: Measures the number of periods since the highest price within the
       lookback period and is calculated as:
       Aroon Up = ((period - periods since highest high) / period) * 100
    
    2. Aroon Down: Measures the number of periods since the lowest price within the
       lookback period and is calculated as:
       Aroon Down = ((period - periods since lowest low) / period) * 100
       
    3. Aroon Oscillator: The difference between Aroon Up and Aroon Down:
       Aroon Oscillator = Aroon Up - Aroon Down
    
    Interpretation:
    
    - Aroon Up/Down range between 0 and 100
    - Aroon Up values close to 100 indicate a strong uptrend
    - Aroon Down values close to 100 indicate a strong downtrend
    - When Aroon Up crosses above Aroon Down, it may signal a bullish trend
    - When Aroon Down crosses above Aroon Up, it may signal a bearish trend
    - The Aroon Oscillator ranges from -100 to 100:
      - Positive values indicate an uptrend
      - Negative values indicate a downtrend
      - Values near zero indicate consolidation or weak trends
    
    Use Cases:
    
    - Trend identification: Determine the direction and strength of the current trend.
    - Trend exhaustion: Identify when a trend is losing momentum before a reversal.
    - Consolidation detection: When both Aroon Up and Down are low, it suggests 
      price consolidation.
    - Breakout confirmation: A strong move in Aroon Up/Down can confirm a price breakout.
    - Crossover signals: When Aroon Up crosses above/below Aroon Down, it may indicate
      a potential trend change.
    """
    high = df[high_col]
    low = df[low_col]
    
    # Create aroon_up and aroon_down series
    aroon_up = pd.Series(index=high.index, dtype=float)
    aroon_down = pd.Series(index=low.index, dtype=float)
    
    # Calculate Aroon indicators for each rolling window
    for i in range(len(high) - period + 1):
        # Get the current window
        high_window = high.iloc[i:i+period]
        low_window = low.iloc[i:i+period]
        
        # Find the highest high and lowest low
        highest_high = high_window.max()
        lowest_low = low_window.min()
        
        # Find the periods since highest high and lowest low
        periods_since_highest = period - 1 - high_window.values.tolist().index(highest_high)
        periods_since_lowest = period - 1 - low_window.values.tolist().index(lowest_low)
        
        # Calculate Aroon Up and Aroon Down
        aroon_up.iloc[i+period-1] = ((period - periods_since_highest) / period) * 100
        aroon_down.iloc[i+period-1] = ((period - periods_since_lowest) / period) * 100
    
    # Calculate Aroon Oscillator
    aroon_oscillator = aroon_up - aroon_down

    df_aroon = pd.DataFrame({
        f'AROON_UP_{period}': aroon_down,
        f'AROON_DOWN_{period}': aroon_up,
        f'AROON_OSCILLATOR_{period}': aroon_oscillator
    })
    df_aroon.index = high.index
    return df_aroon
