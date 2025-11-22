import numpy as np


def emv(df, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Ease of Movement (EMV), an indicator that relates price change
    to volume and shows how easily a price can move up or down. High EMV values
    occur when price moves upward on low volume.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The smoothing period for the EMV. Default is 14.
            - divisor (int): Divisor to scale the EMV values. Default is 10000.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the EMV series and a list of column names.

    The Ease of Movement is calculated as follows:

    1. Calculate Distance Moved (Midpoint Move):
       Distance = ((High + Low) / 2) - ((PrevHigh + PrevLow) / 2)

    2. Calculate Box Ratio:
       Box Ratio = (Volume / divisor) / (High - Low)

    3. Calculate 1-Period EMV:
       EMV_1 = Distance / Box Ratio

    4. Calculate Smoothed EMV:
       EMV = SMA(EMV_1, period)

    Interpretation:
    - Positive EMV: Price rising with ease (Buying pressure).
    - Negative EMV: Price falling with ease (Selling pressure).
    - Near Zero: Heavy volume required to move price, or little price movement.

    Use Cases:
    - Trend Confirmation: Confirm strength of trend (Ease of movement in trend direction).
    - Volume Analysis: Assess the "fuel" behind price moves.
    - Entry Signals: Crossovers of the zero line.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = parameters.get('period', 14)
    divisor = parameters.get('divisor', 10000)
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    volume = df[volume_col]
    
    # Calculate midpoint
    midpoint = (high + low) / 2
    
    # Calculate Distance Moved
    distance_moved = midpoint.diff()
    
    # Calculate Box Ratio (handle division by zero)
    high_low_diff = high - low
    box_ratio = (volume / divisor) / high_low_diff.replace(0, np.nan)
    
    # Calculate 1-Period EMV
    one_period_emv = distance_moved / box_ratio
    one_period_emv = one_period_emv.fillna(0)
    
    # Calculate EMV as simple moving average
    emv_values = one_period_emv.rolling(window=period, min_periods=1).mean()
    
    emv_values.name = f'EMV_{period}'
    columns_list = [emv_values.name]
    return emv_values, columns_list
