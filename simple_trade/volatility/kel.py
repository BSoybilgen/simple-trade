import pandas as pd
import numpy as np
from ..trend.ema import ema
from .atr import atr


def kel(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Keltner Channels, a volatility-based envelope set above and below an exponential moving average.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - ema_window (int): The window for the EMA calculation. Default is 20.
            - atr_window (int): The window for the ATR calculation. Default is 10.
            - atr_multiplier (float): Multiplier for the ATR to set channel width. Default is 2.0.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
    
    Returns:
        tuple: A tuple containing the Keltner Channels DataFrame and a list of column names.
    
    Calculation Steps:
    1. Middle Line:
       Exponential Moving Average (EMA) of the closing price over ema_window.
    2. Average True Range (ATR):
       ATR calculated over atr_window.
    3. Upper Band:
       Middle Line + (ATR * atr_multiplier)
    4. Lower Band:
       Middle Line - (ATR * atr_multiplier)
    
    Interpretation:
    - Price above Upper Band: Strong uptrend, potential overbought.
    - Price below Lower Band: Strong downtrend, potential oversold.
    - Middle Line slope indicates trend direction.
    
    Use Cases:
    - Identifying trend direction.
    - Spotting breakouts (price closing outside channels).
    - Overbought/oversold conditions (mean reversion to middle line).
    - Support and resistance (bands act as dynamic levels).
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    ema_window = int(parameters.get('ema_window', 20))
    atr_window = int(parameters.get('atr_window', 10))
    atr_multiplier = float(parameters.get('atr_multiplier', 2.0))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate the middle line (EMA of close)
    ema_parameters = {'window': ema_window}
    ema_columns = {'close_col': close_col}
    middle_line_series, _ = ema(df, parameters=ema_parameters, columns=ema_columns)
    
    # Calculate ATR for the upper and lower bands
    atr_parameters = {'window': atr_window}
    atr_columns = {'high_col': high_col, 'low_col': low_col, 'close_col': close_col}
    atr_values_series, _ = atr(df, parameters=atr_parameters, columns=atr_columns)
    
    # Calculate the upper and lower bands
    upper_band = middle_line_series + (atr_values_series * atr_multiplier)
    lower_band = middle_line_series - (atr_values_series * atr_multiplier)
    
    # Prepare the result DataFrame
    result = pd.DataFrame({
        f'KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier}': middle_line_series,
        f'KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}': upper_band,
        f'KELT_Lower_{ema_window}_{atr_window}_{atr_multiplier}': lower_band
    }, index=close.index)
    
    columns_list = list(result.columns)
    return result, columns_list
