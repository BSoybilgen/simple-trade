import pandas as pd


def vqi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volatility Quality Index (VQI), an indicator that measures the quality
    of price movements by analyzing the relationship between price changes, volume, and
    volatility to identify genuine trends versus noise.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculations. Default is 9.
            - smooth_period (int): The period for smoothing the VQI. Default is 9.
            - volatility_cutoff (float): Multiplier for ATR to filter noise. Default is 0.1.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the VQI series and a list of column names.

    Calculation Steps:
    1. Calculate True Range (TR) and ATR.
    2. Calculate Price Change:
       Change = Close - Previous Close
    3. Filter Noise:
       If abs(Change) < (volatility_cutoff * ATR), then Change is treated as 0.
    4. Calculate Directional Volume:
       If Change > 0: Move = Change * Volume
       If Change < 0: Move = Change * Volume
       Else: Move = 0
    5. Sum Movements:
       VQI_Raw = Sum(Move, period)
    6. Smooth VQI:
       VQI = EMA(VQI_Raw, smooth_period)

    Interpretation:
    - Positive VQI: Quality uptrend supported by volume.
    - Negative VQI: Quality downtrend supported by volume.
    - Near Zero: Choppy/Noisy market or low volume movements.

    Use Cases:
    - Trend Quality Assessment: Distinguishing between real trends and false moves.
    - Trade Filtering: Avoiding trades in low VQI conditions.
    - Divergence: Spotting disagreements between price and VQI.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = int(parameters.get('period', 9))
    smooth_period = int(parameters.get('smooth_period', 9))
    volatility_cutoff = float(parameters.get('volatility_cutoff', 0.1))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]
    
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR using EMA
    atr = tr.ewm(span=period, adjust=False).mean()
    
    # Calculate price change
    price_change = close.diff()
    
    # Calculate cutoff threshold
    cutoff = volatility_cutoff * atr
    
    # Calculate volume-weighted directional movement
    vqi_raw = pd.Series(0.0, index=close.index)
    
    # Positive movements (above cutoff)
    positive_mask = price_change > cutoff
    vqi_raw[positive_mask] = price_change[positive_mask] * volume[positive_mask]
    
    # Negative movements (below -cutoff)
    negative_mask = price_change < -cutoff
    vqi_raw[negative_mask] = price_change[negative_mask] * volume[negative_mask]
    
    # Sum over period
    vqi_sum = vqi_raw.rolling(window=period).sum()
    
    # Apply smoothing
    vqi_smoothed = vqi_sum.ewm(span=smooth_period, adjust=False).mean()
    
    vqi_smoothed.name = f'VQI_{period}_{smooth_period}'
    columns_list = [vqi_smoothed.name]
    return vqi_smoothed, columns_list
