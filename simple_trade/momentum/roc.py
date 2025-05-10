import pandas as pd
import numpy as np


def roc(df: pd.DataFrame, window: int = 12, close_col: str = 'Close') -> pd.Series:
    """
    Calculates the Rate of Change (ROC), a momentum oscillator that measures the percentage 
    change in price between the current price and the price a specified number of periods ago.

    Args:
        df (pd.DataFrame): The input DataFrame. 
        window (int): The lookback period for the calculation. Default is 12.
        close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        pd.Series: ROC values for the given input series.

    The ROC is calculated using the formula:
    
    ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100
    
    Where n is the specified window.

    Use Cases:

    - Identifying overbought/oversold conditions: Extreme positive values may indicate overbought 
      conditions, while extreme negative values may indicate oversold conditions.
    - Divergence analysis: When price makes a new high or low but ROC doesn't, it may signal 
      a potential reversal.
    - Zero-line crossovers: When ROC crosses above zero, it may signal a buy opportunity; 
      when it crosses below zero, it may signal a sell opportunity.
    - Trend confirmation: Strong positive ROC values confirm an uptrend, while strong negative 
      values confirm a downtrend.
    - Measuring momentum strength: The slope of the ROC line indicates the strength of momentum; 
      a steeper slope indicates stronger momentum.
    """
    series = df[close_col]
    
    # Calculate the Rate of Change
    # ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100
    roc_values = ((series / series.shift(window)) - 1) * 100
    
    roc_values.name = f'ROC_{window}'
    
    return roc_values
