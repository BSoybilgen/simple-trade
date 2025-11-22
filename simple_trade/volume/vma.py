import pandas as pd


def vma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volume Moving Average (VMA), which is a weighted moving average
    that uses volume as the weighting factor. It gives more weight to prices
    accompanied by higher volume.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the VMA series and a list of column names.

    The Volume Moving Average is calculated as follows:

    1. Calculate Weighted Price:
       Weighted Price = Price * Volume

    2. Calculate VMA:
       VMA = Sum(Weighted Price, window) / Sum(Volume, window)

    Interpretation:
    - Rising VMA: Bullish trend supported by volume.
    - Falling VMA: Bearish trend supported by volume.
    - Support/Resistance: VMA often acts as dynamic support or resistance.

    Use Cases:
    - Trend Identification: Identify volume-supported trends.
    - Dynamic Support/Resistance: Use VMA lines for entry/exit points.
    - Filtering: Validate price moves (price above VMA in uptrend).
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
