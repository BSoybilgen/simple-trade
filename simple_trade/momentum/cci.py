import pandas as pd
import numpy as np


def cci(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Commodity Channel Index (CCI), a momentum oscillator used to identify cyclical trends
    and extreme market conditions.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
            - constant (float): The scaling factor used in the CCI formula. Default is 0.015.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the CCI series and a list of column names.

    The CCI is calculated in three steps:

    1. Calculate the Typical Price (TP):
       TP = (High + Low + Close) / 3

    2. Calculate the Simple Moving Average of the Typical Price (SMA(TP)):
       SMA(TP) = n-period SMA of TP

    3. Calculate the Mean Deviation (MD):
       MD = Mean of absolute deviations of TP from SMA(TP)

    4. Calculate the CCI:
       CCI = (TP - SMA(TP)) / (constant * MD)

    The constant (0.015) is used to normalize the CCI to make it comparable across different securities.
    
    Use Cases:

    - Identifying overbought/oversold conditions: Values above +100 suggest overbought conditions,
      while values below -100 suggest oversold conditions.
    - Detecting trend strength: Values consistently above +100 indicate a strong uptrend, while
      values consistently below -100 indicate a strong downtrend.
    - Identifying potential reversals: Divergence between CCI and price can signal potential reversals.
    - Generating trading signals: Crossing above/below zero line or +/-100 thresholds can generate signals.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 20))
    constant = float(parameters.get('constant', 0.015))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]      
    
    # Calculate the Typical Price
    typical_price = (high + low + close) / 3
    
    # Calculate the Simple Moving Average of the Typical Price
    sma_tp = typical_price.rolling(window=window).mean()
    
    # Calculate the Mean Deviation
    mean_deviation = typical_price.rolling(window=window).apply(
        lambda x: np.mean(np.abs(x - np.mean(x)))
    )
    
    # Avoid division by zero
    mean_deviation = mean_deviation.replace(0, np.nan)
    
    # Calculate the CCI
    cci = (typical_price - sma_tp) / (constant * mean_deviation)

    cci.name = f'CCI_{window}_{constant}'
    columns_list = [cci.name]
    return cci, columns_list
