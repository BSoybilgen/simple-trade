import pandas as pd
import numpy as np


def vma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volume Moving Average (VMA), which is a weighted moving average
    that uses volume as the weighting factor.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.
    
    Returns:
        tuple: A tuple containing the VMA series and a list of column names.
    
    VMA gives more weight to price moves accompanied by higher volume, making it
    more responsive to significant market movements than a simple moving average.
    
    The formula is:
    VMA = Σ(Price * Volume) / Σ(Volume), calculated over the specified window.
    
    Use Cases:
    
    - Trend identification: VMA can be used similarly to other moving averages to
      identify trends, but with more emphasis on volume-supported price movements.
    - Dynamic support/resistance: VMA can act as support in uptrends and resistance
      in downtrends.
    - Entry/exit signals: Crossovers between price and VMA or between multiple VMAs
      with different periods can generate trading signals.
    - Volume-validated price movement: VMA filters out price movements that occur on
      low volume, focusing on more significant market activity.
    - Divergence analysis: Comparing VMA to other moving averages can highlight periods
      where price moves are or aren't supported by volume.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 20))
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')

    close = df[close_col]
    volume = df[volume_col]
    
    # Calculate the volume-weighted price
    weighted_price = close * volume
    
    # Calculate the VMA using rolling windows
    # For each window, sum(price * volume) / sum(volume)
    vma_values = weighted_price.rolling(window=window).sum() / volume.rolling(window=window).sum()
    vma_values.name = f'VMA_{window}'
    columns_list = [vma_values.name]
    return vma_values, columns_list
