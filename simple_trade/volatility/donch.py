import pandas as pd
import numpy as np


def donchian_channels(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Calculates Donchian Channels, a volatility indicator that plots the highest high and lowest low
    over a specified period.
    
    Args:
        high (pd.Series): The high prices of the period.
        low (pd.Series): The low prices of the period.
        window (int): The lookback period for the calculation. Default is 20.
    
    Returns:
        pd.DataFrame: A DataFrame containing the upper band (highest high), middle band (mean of upper and lower),
                     and lower band (lowest low).
    
    Donchian Channels consist of three lines:
    
    1. Upper Band: The highest high over the specified period.
    2. Lower Band: The lowest low over the specified period.
    3. Middle Band: The average of the upper and lower bands.
    
    Use Cases:
    
    - Breakout trading: A break above the upper band or below the lower band can signal a potential breakout.
    - Trend identification: The direction of the middle band can indicate the overall trend.
    - Support and resistance: The upper and lower bands can serve as dynamic resistance and support levels.
    - Range definition: The bands clearly define the trading range over the specified period.
    - Volatility measurement: The width between the upper and lower bands can indicate market volatility.
    
    Notably, Donchian Channels are a key component of the original "Turtle Trading" system, a trend-following
    strategy developed by Richard Dennis and William Eckhardt in the 1980s.
    """
    # Make sure inputs have the same index
    high = high.copy()
    low = low.copy()
    
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
    
    return result
