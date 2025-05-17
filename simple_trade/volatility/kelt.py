import pandas as pd
import numpy as np
from ..trend.ema import ema
from .atr import atr


def keltner_channels(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> pd.DataFrame:
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
        pd.DataFrame: A DataFrame containing the middle line (EMA), upper band, and lower band.
    
    Keltner Channels consist of three lines:
    
    1. Middle Line: An Exponential Moving Average (EMA) of the typical price or closing price.
    2. Upper Band: EMA + (ATR * multiplier)
    3. Lower Band: EMA - (ATR * multiplier)
    
    The ATR multiplier determines the width of the channels. Higher multipliers create wider channels.
    
    Use Cases:
    
    - Identifying trend direction: Price consistently above or below the middle line can confirm trend direction.
    - Spotting breakouts: Price moving outside the channels may signal a potential breakout.
    - Overbought/oversold conditions: Price reaching the upper band may be overbought, while price reaching 
      the lower band may be oversold.
    - Range identification: Narrow channels suggest consolidation, while wide channels indicate volatility.
    - Support and resistance: The upper and lower bands can act as dynamic support and resistance levels.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    ema_window = parameters.get('ema_window', 20)
    atr_window = parameters.get('atr_window', 10)
    atr_multiplier = parameters.get('atr_multiplier', 2.0)
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate the middle line (EMA of close)
    ema_parameters = {'window': ema_window}
    ema_columns = {'close_col': close_col}
    middle_line = ema(df, parameters=ema_parameters, columns=ema_columns)
    
    # Calculate ATR for the upper and lower bands
    atr_parameters = {'window': atr_window}
    atr_columns = {'high_col': high_col, 'low_col': low_col, 'close_col': close_col}
    atr_values = atr(df, parameters=atr_parameters, columns=atr_columns)
    
    # Calculate the upper and lower bands
    upper_band = middle_line + (atr_values * atr_multiplier)
    lower_band = middle_line - (atr_values * atr_multiplier)
    
    # Prepare the result DataFrame
    result = pd.DataFrame({
        f'KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier}': middle_line,
        f'KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}': upper_band,
        f'KELT_Lower_{ema_window}_{atr_window}_{atr_multiplier}': lower_band
    }, index=close.index)
    
    return result
