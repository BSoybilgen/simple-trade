import pandas as pd
import numpy as np


def roc(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Rate of Change (ROC), a momentum oscillator that measures the percentage 
    change in price between the current price and the price a specified number of periods ago.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 12.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the ROC series and a list of column names.

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
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 12))
    close_col = columns.get('close_col', 'Close')
    
    series = df[close_col]
    
    # Calculate the Rate of Change
    # ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100
    roc_values = ((series / series.shift(window)) - 1) * 100
    
    roc_values.name = f'ROC_{window}'
    columns_list = [roc_values.name]
    return roc_values, columns_list
