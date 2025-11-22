import numpy as np
import pandas as pd


def mgd(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the McGinley Dynamic (MGD) indicator.
    The McGinley Dynamic is a moving average that automatically adjusts its speed
    based on market volatility. It speeds up in fast markets and slows down in
    ranging markets, minimizing whipsaws and price separation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the McGinley Dynamic series and a list of column names.

    The McGinley Dynamic is calculated as follows:

    1. Calculate McGinley Dynamic (MD):
       MD[i] = MD[i-1] + (Price[i] - MD[i-1]) / (N * (Price[i] / MD[i-1])^4)
       where N is the period parameter (window).

    Interpretation:
    - It looks like a moving average but tracks the price much better.
    - It avoids the lag of SMA and the whipsaws of EMA.

    Use Cases:
    - Trend Following: A reliable trend line that hugs the price action.
    - Dynamic Support/Resistance: Often acts as a strong support/resistance level.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    series = df[close_col].copy()
    
    # Initialize McGinley Dynamic with SMA
    md = pd.Series(index=series.index, dtype=float)
    md.iloc[:window] = series.iloc[:window].rolling(window=window).mean()
    
    # Calculate McGinley Dynamic
    for i in range(window, len(series)):
        if pd.notna(md.iloc[i-1]) and md.iloc[i-1] != 0:
            ratio = series.iloc[i] / md.iloc[i-1]
            # Prevent extreme values
            ratio = np.clip(ratio, 0.1, 10.0)
            k = (series.iloc[i] - md.iloc[i-1]) / (window * (ratio ** 4))
            md.iloc[i] = md.iloc[i-1] + k
        else:
            md.iloc[i] = series.iloc[i]
    
    md.name = f'MGD_{window}'
    
    columns_list = [md.name]
    return md, columns_list
