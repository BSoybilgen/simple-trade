import pandas as pd

def voo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volume Oscillator (VO), which displays the difference between
    two moving averages of volume. It measures volume trends using points (absolute difference)
    rather than percentage.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - fast_period (int): The fast SMA period. Default is 5.
            - slow_period (int): The slow SMA period. Default is 10.
        columns (dict, optional): Dictionary containing column name mappings:
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the VO series and a list of column names.

    The Volume Oscillator is calculated as follows:

    1. Calculate Fast SMA:
       Fast SMA = SMA(Volume, fast_period)

    2. Calculate Slow SMA:
       Slow SMA = SMA(Volume, slow_period)

    3. Calculate VO:
       VO = Fast SMA - Slow SMA

    Interpretation:
    - Positive VO: Volume is increasing (Short-term volume > Long-term volume).
    - Negative VO: Volume is decreasing (Short-term volume < Long-term volume).
    - Trend Strength: Rising VO indicates increasing market participation.

    Use Cases:
    - Volume Trend Analysis: Confirming the strength of price moves.
    - Breakout Confirmation: VO spikes often accompany valid breakouts.
    - Exhaustion: Extreme VO readings may signal trend exhaustion.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    fast_period = parameters.get('fast_period', 5)
    slow_period = parameters.get('slow_period', 10)
    volume_col = columns.get('volume_col', 'Volume')
    
    volume = df[volume_col]
    
    # Calculate SMAs
    fast_sma = volume.rolling(window=fast_period).mean()
    slow_sma = volume.rolling(window=slow_period).mean()
    
    # Calculate VO
    vo_values = fast_sma - slow_sma
    
    vo_values.name = f'VO_{fast_period}_{slow_period}'
    columns_list = [vo_values.name]
    return vo_values, columns_list
