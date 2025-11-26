import pandas as pd


def tsv(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the True Strength Index (TSI) Volatility, which applies the TSI momentum
    indicator formula to volatility measures (ATR or standard deviation) to create a
    double-smoothed volatility momentum indicator.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - atr_period (int): Period for ATR calculation. Default is 14.
            - long_period (int): Long smoothing period. Default is 25.
            - short_period (int): Short smoothing period. Default is 13.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the TSI Volatility series and a list of column names.

    Calculation Steps:
    1. Calculate ATR (Average True Range).
    2. Calculate ATR Momentum:
       Momentum = ATR - ATR(previous)
    3. Double Smooth Momentum:
       Smooth1 = EMA(Momentum, long_period)
       Smooth2 = EMA(Smooth1, short_period)
    4. Double Smooth Absolute Momentum:
       AbsSmooth1 = EMA(|Momentum|, long_period)
       AbsSmooth2 = EMA(AbsSmooth1, short_period)
    5. Calculate TSI:
       TSI = 100 * (Smooth2 / AbsSmooth2)

    Interpretation:
    - Positive TSI: Rising volatility (Momentum Up).
    - Negative TSI: Falling volatility (Momentum Down).
    - Zero Line Crossovers: Trend changes in volatility.

    Use Cases:
    - Volatility trend identification.
    - Divergence detection (price vs volatility momentum).
    - Filtering noise with double smoothing.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    atr_period = int(parameters.get('atr_period', 14))
    long_period = int(parameters.get('long_period', 25))
    short_period = int(parameters.get('short_period', 13))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.ewm(span=atr_period, adjust=False).mean()
    
    # Calculate ATR momentum (change)
    atr_momentum = atr.diff()
    
    # Double smoothing of momentum
    smooth1 = atr_momentum.ewm(span=long_period, adjust=False).mean()
    smooth2 = smooth1.ewm(span=short_period, adjust=False).mean()
    
    # Double smoothing of absolute momentum
    abs_momentum = atr_momentum.abs()
    abs_smooth1 = abs_momentum.ewm(span=long_period, adjust=False).mean()
    abs_smooth2 = abs_smooth1.ewm(span=short_period, adjust=False).mean()
    
    # Calculate TSI
    tsi_values = 100 * (smooth2 / abs_smooth2)
    tsi_values = tsi_values.fillna(0)
    
    tsi_values.name = f'TSI_VOL_{atr_period}_{long_period}_{short_period}'
    columns_list = [tsi_values.name]
    return tsi_values, columns_list
