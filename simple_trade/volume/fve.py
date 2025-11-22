import pandas as pd
import numpy as np


def fve(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Finite Volume Elements (FVE), a money flow indicator that resolves
    volatility by separating volume into "bullish" and "bearish" components based on
    intra-period price action and a volatility threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period. Default is 22.
            - factor (float): The cutoff factor (percent). Default is 0.3 (0.3%).
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the FVE series and a list of column names.

    The Finite Volume Elements is calculated as follows:

    1. Calculate Typical Price (TP):
       TP = (High + Low + Close) / 3

    2. Calculate Price Change:
       Change = TP - TP(previous)

    3. Determine Cutoff Threshold:
       Cutoff = (factor / 100) * Close

    4. Assign Volume Direction:
       - If Change > Cutoff: Bullish Volume (+Volume)
       - If Change < -Cutoff: Bearish Volume (-Volume)
       - Otherwise: Neutral Volume (0)

    5. Calculate FVE:
       FVE = (Sum(Volume Direction, period) / Sum(Volume, period)) * 100

    Interpretation:
    - FVE > 0: Bullish money flow.
    - FVE < 0: Bearish money flow.
    - Rising FVE: Buying pressure increasing.

    Use Cases:
    - Trend Confirmation: Confirm price trends with money flow.
    - Divergence: Spot reversals when Price and FVE diverge.
    - Breakout Validation: FVE crossing zero can signal new trends.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = parameters.get('period', 22)
    factor = parameters.get('factor', 0.3)
    
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
    
    # 2. Price Change
    tp_change = tp.diff()
    
    # 3. Cutoff
    cutoff = (factor / 100.0) * close
    
    # 4. Volume Direction
    vol_direction = pd.Series(0.0, index=df.index)
    
    # Bullish
    mask_bull = tp_change > cutoff
    vol_direction[mask_bull] = volume[mask_bull]
    
    # Bearish
    mask_bear = tp_change < -cutoff
    vol_direction[mask_bear] = -volume[mask_bear]
    
    # 5. FVE Calculation
    vol_sum = volume.rolling(window=period).sum()
    dir_sum = vol_direction.rolling(window=period).sum()
    
    fve_values = (dir_sum / vol_sum.replace(0, np.nan)) * 100
    
    fve_values.name = f'FVE_{period}'
    columns_list = [fve_values.name]
    return fve_values, columns_list
