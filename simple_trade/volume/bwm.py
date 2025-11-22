import numpy as np


def bwm(df, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Bill Williams Market Facilitation Index (BW MFI), which measures
    the efficiency of price movement by analyzing the change in price per unit of volume.
    It helps determine the willingness of the market to move the price.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters.
            No parameters are used.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the BW MFI series and a list of column names.

    The Market Facilitation Index is calculated as follows:

    1. Calculate Range:
       Range = High - Low

    2. Calculate BW MFI:
       BW MFI = Range / Volume

    Interpretation (when combined with Volume):
    - Green (MFI Up, Vol Up): Strong trend, increasing participation.
    - Fade (MFI Down, Vol Down): Market losing interest, potential reversal.
    - Fake (MFI Up, Vol Down): Price moving without volume support (speculative).
    - Squat (MFI Down, Vol Up): High volume but little movement, battle between bulls/bears.

    Use Cases:
    - Trend Strength: Identify if price moves are supported by volume.
    - Reversal Warning: "Squat" bars often precede reversals.
    - Filtering: Avoid trading during "Fade" conditions.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    volume = df[volume_col]
    
    # Calculate MFI
    # Handle division by zero by replacing 0 volume with NaN (or a very small number if preferred)
    # Using NaN results in NaN MFI, which is safer than Infinity
    mfi_values = (high - low) / volume.replace(0, np.nan)
    mfi_values = mfi_values.fillna(0)
    
    mfi_values.name = 'BWMFI'
    columns_list = [mfi_values.name]
    return mfi_values, columns_list
