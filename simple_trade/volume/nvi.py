import pandas as pd


def nvi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Negative Volume Index (NVI), an indicator that tracks price changes
    on days when volume decreases from the previous day. It is based on the premise
    that "smart money" trades on low volume days.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - initial_value (float): The starting value for NVI. Default is 1000.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the NVI series and a list of column names.

    The Negative Volume Index is calculated as follows:

    1. Initialize NVI:
       Start with an arbitrary value (e.g., 1000).

    2. Iterate through each period:
       - If Current Volume < Previous Volume:
         NVI = Previous NVI + (Previous NVI * Price ROC)
         where Price ROC = (Close - Previous Close) / Previous Close
       - If Current Volume >= Previous Volume:
         NVI = Previous NVI (No Change)

    Interpretation:
    - Rising NVI: Smart money is accumulating.
    - Falling NVI: Smart money is distributing.
    - Above EMA: Bullish trend.
    - Below EMA: Bearish trend.

    Use Cases:
    - Smart Money Tracking: identifying institutional accumulation/distribution.
    - Trend Identification: Long-term trend direction.
    - Confirmation: Often used with Positive Volume Index (PVI).
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
    
    # Initialize NVI series
    nvi_values = pd.Series(index=df.index, dtype=float)
    nvi_values.iloc[0] = initial_value
    
    # Calculate NVI
    for i in range(1, len(df)):
        if volume.iloc[i] < volume.iloc[i-1]:
            # Volume decreased - update NVI based on price change
            price_change_pct = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            nvi_values.iloc[i] = nvi_values.iloc[i-1] + (nvi_values.iloc[i-1] * price_change_pct)
        else:
            # Volume increased or stayed same - NVI unchanged
            nvi_values.iloc[i] = nvi_values.iloc[i-1]
    
    nvi_values.name = 'NVI'
    columns_list = [nvi_values.name]
    return nvi_values, columns_list
