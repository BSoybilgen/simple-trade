import pandas as pd
import numpy as np


def adx(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Average Directional Index (ADX) along with the Positive
    Directional Indicator (+DI) and Negative Directional Indicator (-DI).
    The ADX is a technical indicator used to measure the strength of a trend.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the ADX calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the ADX DataFrame (ADX, +DI, -DI) and a list of column names.

    Calculation Steps:
    1. Calculate True Range (TR):
       TR = Max(High-Low, Abs(High-PrevClose), Abs(Low-PrevClose))

    2. Calculate Directional Movements (+DM, -DM):
       +DM = Current High - Previous High (if > 0 and > -DM, else 0)
       -DM = Previous Low - Current Low (if > 0 and > +DM, else 0)

    3. Smooth TR, +DM, and -DM (using rolling mean/RMA):
       ATR = Smooth(TR)
       +Smoothed = Smooth(+DM)
       -Smoothed = Smooth(-DM)

    4. Calculate Directional Indicators (+DI, -DI):
       +DI = 100 * (+Smoothed / ATR)
       -DI = 100 * (-Smoothed / ATR)

    5. Calculate Directional Index (DX):
       DX = 100 * Abs(+DI - -DI) / (+DI + -DI)

    6. Calculate ADX:
       ADX = Smooth(DX)

    Interpretation:
    - ADX > 25: Strong trend.
    - ADX < 20: Weak trend or non-trending market.
    - +DI > -DI: Bullish trend.
    - -DI > +DI: Bearish trend.

    Use Cases:
    - Identifying trend strength: The ADX can be used to determine whether a
      trend is strong or weak.
    - Identifying trend direction: The +DI and -DI can be used to determine
      the direction of the trend.
    - Generating buy and sell signals: Crossovers of +DI and -DI.
    - Filtering: Using ADX to filter out trades in sideways markets.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 14))

    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_ = dx.rolling(window=window).mean()
    df_adx = pd.DataFrame({
        f'ADX_{window}': adx_,
        f'+DI_{window}': plus_di,
        f'-DI_{window}': minus_di
    })
    df_adx.index = close.index

    columns = list(df_adx.columns)

    return df_adx, columns