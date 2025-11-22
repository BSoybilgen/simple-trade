import pandas as pd
import numpy as np


def vwa(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volume Weighted Average Price (VWAP), a trading benchmark
    that gives the average price a security has traded at throughout the day (or dataset),
    based on both volume and price.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters.
            No parameters are used.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the VWAP series and a list of column names.

    The Volume Weighted Average Price is calculated as follows:

    1. Calculate Typical Price (TP):
       TP = (High + Low + Close) / 3

    2. Calculate TPV (Typical Price * Volume):
       TPV = TP * Volume

    3. Calculate Cumulative TPV:
       CumTPV = CumulativeSum(TPV)

    4. Calculate Cumulative Volume:
       CumVol = CumulativeSum(Volume)

    5. Calculate VWAP:
       VWAP = CumTPV / CumVol

    Interpretation:
    - Price > VWAP: Bullish sentiment (Buyers in control).
    - Price < VWAP: Bearish sentiment (Sellers in control).
    - Benchmark: Acts as a measure of "fair value" for the period.

    Use Cases:
    - Intraday Trading: Assessing if price is expensive or cheap relative to the day's average.
    - Trade Execution: Benchmarking trade fills.
    - Support/Resistance: VWAP often acts as a magnet or dynamic level.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]
    
    # Calculate Typical Price
    typical_price = (high + low + close) / 3
    
    # Calculate cumulative TPV (Typical Price * Volume)
    cumulative_tpv = (typical_price * volume).cumsum()
    
    # Calculate cumulative volume
    cumulative_volume = volume.cumsum()
    
    # Calculate VWAP (handle division by zero)
    vwap_values = cumulative_tpv / cumulative_volume.replace(0, np.nan)
    vwap_values = vwap_values.fillna(method='ffill').fillna(0)
    
    vwap_values.name = 'VWAP'
    columns_list = [vwap_values.name]
    return vwap_values, columns_list
