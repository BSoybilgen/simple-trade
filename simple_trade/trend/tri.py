import pandas as pd
import numpy as np


def tri(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the TRIX (Triple Exponential Average) indicator.
    TRIX is a momentum oscillator that displays the percent rate of change of a triple
    exponentially smoothed moving average. It oscillates around a zero line and can be
    used to identify overbought/oversold conditions, divergences, and trend direction.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the EMA. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the TRIX DataFrame and a list of column names.

    The TRIX is calculated as follows:

    1. Calculate Single-Smoothed EMA:
       EMA1 = EMA(Close, window)

    2. Calculate Double-Smoothed EMA:
       EMA2 = EMA(EMA1, window)

    3. Calculate Triple-Smoothed EMA:
       EMA3 = EMA(EMA2, window)

    4. Calculate TRIX:
       TRIX = 100 * (EMA3 - Previous EMA3) / Previous EMA3

    5. Calculate Signal Line:
       Signal = EMA(TRIX, 9)

    Interpretation:
    - Zero Line Crossover: Crossing above zero is bullish; below zero is bearish.
    - Signal Line Crossover: TRIX crossing above Signal is bullish; below is bearish.
    - Divergences: Price making new highs while TRIX fails to do so indicates a potential reversal.

    Use Cases:
    - Momentum Measurement: Gauging the rate of change of the triple smoothed average.
    - Trend Reversals: Identifying early signs of trend shifts via divergences.
    - Filtering: TRIX filters out insignificant price movements better than standard rate-of-change.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 14))

    series = df[close_col]
    # Step 1: Calculate the single-smoothed EMA
    ema1 = series.ewm(span=window, adjust=False).mean()
    
    # Step 2: Calculate the double-smoothed EMA
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    
    # Step 3: Calculate the triple-smoothed EMA
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    
    # Step 4: Calculate the 1-period percent rate of change of the triple-smoothed EMA
    trix_line = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    
    # Calculate signal line (9-period EMA of TRIX)
    signal_line = trix_line.ewm(span=9, adjust=False).mean()
    
    # Create result DataFrame
    df_trix = pd.DataFrame({
        f'TRIX_{window}': trix_line,
        f'TRIX_SIGNAL_{window}': signal_line
    })
    df_trix.index = series.index

    columns_list = list(df_trix.columns)
    return df_trix, columns_list
