import pandas as pd


def pvi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Positive Volume Index (PVI), an indicator that tracks price changes
    on days when volume increases from the previous day. It is based on the premise
    that the "crowd" or uninformed investors trade on high volume days.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - initial_value (float): The starting value for PVI. Default is 1000.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the PVI series and a list of column names.

    The Positive Volume Index is calculated as follows:

    1. Initialize PVI:
       Start with an arbitrary value (e.g., 1000).

    2. Iterate through each period:
       - If Current Volume > Previous Volume:
         PVI = Previous PVI + (Previous PVI * Price ROC)
         where Price ROC = (Close - Previous Close) / Previous Close
       - If Current Volume <= Previous Volume:
         PVI = Previous PVI (No Change)

    Interpretation:
    - Rising PVI: Crowd is buying (Bullish sentiment among general public).
    - Falling PVI: Crowd is selling (Bearish sentiment among general public).
    - Above EMA: Bullish trend.
    - Below EMA: Bearish trend.

    Use Cases:
    - Sentiment Analysis: Gauge the activity of uninformed investors.
    - Trend Confirmation: Confirm trends driven by high volume.
    - Market Phase Identification: Used with NVI to identify Bull/Bear markets.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    initial_value = parameters.get('initial_value', 1000)
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    close = df[close_col]
    volume = df[volume_col]
    
    # Initialize PVI series
    pvi_values = pd.Series(index=df.index, dtype=float)
    pvi_values.iloc[0] = initial_value
    
    # Calculate PVI
    for i in range(1, len(df)):
        if volume.iloc[i] > volume.iloc[i-1]:
            # Volume increased - update PVI based on price change
            price_change_pct = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            pvi_values.iloc[i] = pvi_values.iloc[i-1] + (pvi_values.iloc[i-1] * price_change_pct)
        else:
            # Volume decreased or stayed same - PVI unchanged
            pvi_values.iloc[i] = pvi_values.iloc[i-1]
    
    pvi_values.name = 'PVI'
    columns_list = [pvi_values.name]
    return pvi_values, columns_list
