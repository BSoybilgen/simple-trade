import pandas as pd
import numpy as np


def vfi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volume Flow Indicator (VFI), a long-term trend-following indicator
    that uses the rate of change of price to calculate the direction of volume flow.
    It is based on On-Balance Volume (OBV) but attempts to reduce noise.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for VFI calculation. Default is 130.
            - coef (float): The coefficient for cutoff calculation. Default is 0.2.
            - vcoef (float): The volume clamp coefficient. Default is 2.5.
            - smoothing_period (int): Period for EMA smoothing. Default is 3.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the VFI series and a list of column names.

    The Volume Flow Indicator is calculated as follows:

    1. Calculate Typical Price (TP):
       TP = (High + Low + Close) / 3

    2. Calculate Log Return of TP:
       Change = ln(TP) - ln(TP_prev)

    3. Calculate Cutoff:
       Cutoff = coef * StdDev(Change, 30) * Close

    4. Calculate Volume Direction:
       If (TP - TP_prev) > Cutoff: Use +Volume (Clamped)
       If (TP - TP_prev) < -Cutoff: Use -Volume (Clamped)
       Else: 0

    5. Calculate VFI:
       VFI = Sum(Volume Direction, period) / AvgVolume(period)

    6. Smooth VFI:
       VFI = EMA(VFI, smoothing_period)

    Interpretation:
    - VFI > 0: Bullish money flow.
    - VFI < 0: Bearish money flow.
    - Zero Crossing: Trend reversal signal.

    Use Cases:
    - Long-term Trend Identification.
    - Reversal Detection (Zero Crossings).
    - Divergence Analysis.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = parameters.get('period', 130)
    coef = parameters.get('coef', 0.2)
    vcoef = parameters.get('vcoef', 2.5)
    smoothing_period = parameters.get('smoothing_period', 3)
    
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]
    
    # 1. Typical Price
    tp = (high + low + close) / 3
    
    # 2. Inter-day change (log return of typical price)
    inter = np.log(tp) - np.log(tp.shift(1))
    
    # 3. Cutoff calculation
    vinter = inter.rolling(window=30).std()
    cutoff = coef * vinter * close
    
    # 4. Average Volume
    vave = volume.rolling(window=period).mean()
    
    # 5. Clamp Volume
    vmax = vave * vcoef
    vc = volume.copy()
    vc = pd.DataFrame({'vol': vc, 'max': vmax}).min(axis=1)
    
    # 6. Volume Direction
    mf = tp.diff()
    vfi_val = pd.Series(0.0, index=df.index)
    
    # If mf > cutoff, use +volume
    vfi_val[mf > cutoff] = vc[mf > cutoff]
    # If mf < -cutoff, use -volume
    vfi_val[mf < -cutoff] = -vc[mf < -cutoff]
    
    # 7. VFI Calculation (Sum over period / VAVE)
    # Note: We fill NaN with 0 to handle the beginning of the series
    vfi_sum = vfi_val.rolling(window=period).sum()
    
    # Handle division by zero
    vfi_raw = vfi_sum / vave.replace(0, np.nan)
    
    # 8. Smoothing
    vfi_final = vfi_raw.ewm(span=smoothing_period, adjust=False).mean()
    
    vfi_final.name = f'VFI_{period}'
    columns_list = [vfi_final.name]
    return vfi_final, columns_list
