import pandas as pd
import numpy as np
from ..trend.ema import ema
from .atr import atr


def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, 
                     ema_window: int = 20, atr_window: int = 10, 
                     atr_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Calculates Keltner Channels, a volatility-based envelope set above and below an exponential moving average.
    
    Args:
        high (pd.Series): The high prices of the period.
        low (pd.Series): The low prices of the period.
        close (pd.Series): The closing prices of the period.
        ema_window (int): The window for the EMA calculation (middle line). Default is 20.
        atr_window (int): The window for the ATR calculation. Default is 10.
        atr_multiplier (float): Multiplier for the ATR to set channel width. Default is 2.0.
    
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
    # Make sure all inputs have the same index
    high = high.copy()
    low = low.copy()
    close = close.copy()
    
    # Calculate the middle line (EMA of close)
    middle_line = ema(close, window=ema_window)
    
    # Calculate ATR for the upper and lower bands
    atr_values = atr(high, low, close, window=atr_window)
    
    # Calculate the upper and lower bands
    upper_band = middle_line + (atr_values * atr_multiplier)
    lower_band = middle_line - (atr_values * atr_multiplier)
    
    # Prepare the result DataFrame
    result = pd.DataFrame({
        f'KELT_Middle_{ema_window}': middle_line,
        f'KELT_Upper_{ema_window}': upper_band,
        f'KELT_Lower_{ema_window}': lower_band
    }, index=close.index)
    
    return result
