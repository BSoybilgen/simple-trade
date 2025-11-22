import pandas as pd
import numpy as np


def don(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Donchian Channels, a volatility indicator that plots the highest high and lowest low
    over a specified period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Donchian Channels DataFrame and a list of column names.

    Calculation Steps:
    1. Upper Band:
       Highest High over the specified window.
    2. Lower Band:
       Lowest Low over the specified window.
    3. Middle Band:
       (Upper Band + Lower Band) / 2

    Interpretation:
    - Price breaking Upper Band: Bullish breakout.
    - Price breaking Lower Band: Bearish breakout.
    - Middle Band direction: Indicates overall trend.

    Use Cases:
    - Breakout trading: Basis of the "Turtle Trading" system.
    - Trend identification: Middle band slope.
    - Support and resistance: Dynamic S/R levels.
    - Volatility measurement: Width between bands.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 20))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    
    high = df[high_col]
    low = df[low_col]
    
    # Calculate the upper and lower bands
    upper_band = high.rolling(window=window).max()
    lower_band = low.rolling(window=window).min()
    
    # Calculate the middle band
    middle_band = (upper_band + lower_band) / 2
    
    # Prepare the result DataFrame
    result = pd.DataFrame({
        f'DONCH_Upper_{window}': upper_band,
        f'DONCH_Middle_{window}': middle_band,
        f'DONCH_Lower_{window}': lower_band
    }, index=high.index)
    
    columns_list = list(result.columns)
    return result, columns_list
