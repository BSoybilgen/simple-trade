import pandas as pd


def obv(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the On-Balance Volume (OBV), a volume-based momentum indicator that
    relates volume flow to price changes. It measures buying and selling pressure
    as a cumulative indicator that adds volume on up days and subtracts it on down days.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters.
            No parameters are used.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the OBV series and a list of column names.

    The On-Balance Volume is calculated as follows:

    1. Determine Price Direction:
       If Close > Previous Close: Direction = +1
       If Close < Previous Close: Direction = -1
       If Close = Previous Close: Direction = 0

    2. Calculate OBV:
       OBV = Previous OBV + (Volume * Direction)

    Interpretation:
    - Rising OBV: Buying pressure (Accumulation).
    - Falling OBV: Selling pressure (Distribution).
    - Trend Confirmation: OBV should move in the direction of the price trend.

    Use Cases:
    - Trend Confirmation: Confirm the strength of a trend.
    - Divergence Detection: Divergences between Price and OBV often precede reversals.
    - Breakout Validation: Rising OBV during consolidation can signal a breakout.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    close = df[close_col]
    volume = df[volume_col]

    # Calculate the daily price change direction
    # 1 for price up, -1 for price down, 0 for unchanged
    price_direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # First OBV value is equal to the first period's volume
    obv_values = pd.Series(index=close.index, dtype=float)
    obv_values.iloc[0] = volume.iloc[0]
    
    # Cumulative sum of volume multiplied by price direction
    for i in range(1, len(close)):
        obv_values.iloc[i] = obv_values.iloc[i-1] + (volume.iloc[i] * price_direction.iloc[i])
    
    obv_values.name = 'OBV'
    columns_list = [obv_values.name]
    return obv_values, columns_list
