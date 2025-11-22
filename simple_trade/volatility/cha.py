import pandas as pd
import numpy as np
from ..trend.ema import ema


def cha(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Chaikin Volatility (CV) indicator, which measures volatility by 
    calculating the rate of change of the high-low price range.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - ema_window (int): The window for calculating the EMA of the high-low range. Default is 10.
            - roc_window (int): The window for calculating the rate of change. Default is 10.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Chaikin Volatility series and a list of column names.

    The Chaikin Volatility is calculated as follows:

    1. Calculate Daily Range:
       Range = High - Low

    2. Calculate EMA of Range:
       RangeEMA = EMA(Range, ema_window)

    3. Calculate Rate of Change (ROC):
       CV = ((RangeEMA - RangeEMA(roc_window periods ago)) / RangeEMA(roc_window periods ago)) * 100

    Interpretation:
    - Higher values indicate increasing volatility (range expansion).
    - Lower values indicate decreasing volatility (range contraction).
    - Peaks in CV often correlate with market tops or bottoms.

    Use Cases:
    - Volatility measurement: Identifies periods of increasing or decreasing volatility.
    - Market turning points: Rising volatility often precedes tops; falling volatility often precedes bottoms.
    - Breakout confirmation: Sharp increases in volatility can confirm breakouts.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    ema_window = int(parameters.get('ema_window', 10))
    roc_window = int(parameters.get('roc_window', 10))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    
    high = df[high_col]
    low = df[low_col]
    
    # Calculate the daily high-low range
    hl_range = high - low
    df = pd.DataFrame(hl_range, columns=['Close'])

    # Calculate the EMA of the high-low range
    # Create parameters for ema function
    ema_parameters = {'window': ema_window}
    ema_columns = {'close_col': 'Close'}
    range_ema_series, _ = ema(df, parameters=ema_parameters, columns=ema_columns)
    
    # Calculate the percentage rate of change over roc_window days
    # (Current EMA - EMA roc_window days ago) / (EMA roc_window days ago) * 100
    roc = ((range_ema_series - range_ema_series.shift(roc_window)) / range_ema_series.shift(roc_window)) * 100
    roc.name = f'CHAIK_{ema_window}_{roc_window}'
    columns_list = [roc.name]
    return roc, columns_list
