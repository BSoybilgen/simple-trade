import pandas as pd
import numpy as np
from ..trend.ema import ema
from .atr import atr


def keltner_channels(df: pd.DataFrame, 
                     ema_window: int = 20, atr_window: int = 10, 
                     atr_multiplier: float = 2.0,
                     high_col: str = 'High', low_col: str = 'Low', close_col: str = 'Close'
                     ) -> pd.DataFrame:
    """
    Calculates Keltner Channels, a volatility-based envelope set above and below an exponential moving average.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        ema_window (int): The window for the EMA calculation (middle line). Default is 20.
        atr_window (int): The window for the ATR calculation. Default is 10.
        atr_multiplier (float): Multiplier for the ATR to set channel width. Default is 2.0.
        high_col (str): The column name for high prices. Default is 'High'.
        low_col (str): The column name for low prices. Default is 'Low'.
        close_col (str): The column name for closing prices. Default is 'Close'.
    
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
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate the middle line (EMA of close)
    middle_line = ema(df, window=ema_window, close_col=close_col)
    
    # Calculate ATR for the upper and lower bands
    atr_values = atr(df, window=atr_window, high_col=high_col, low_col=low_col, close_col=close_col)
    
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
