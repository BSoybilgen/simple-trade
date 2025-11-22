import pandas as pd


def awo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Awesome Oscillator (AO), a momentum indicator used to measure market momentum.
    It calculates the difference between a 34-period and 5-period Simple Moving Average 
    applied to the median price (High+Low)/2.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - fast_window (int): The period for the fast SMA. Default is 5.
            - slow_window (int): The period for the slow SMA. Default is 34.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Awesome Oscillator series and a list of column names.

    The Awesome Oscillator is calculated in the following steps:

    1. Calculate the Median Price:
       Median Price = (High + Low) / 2

    2. Calculate the Fast Simple Moving Average (SMA):
       SMA Fast = SMA(Median Price, fast_window)

    3. Calculate the Slow Simple Moving Average (SMA):
       SMA Slow = SMA(Median Price, slow_window)

    4. Calculate the Awesome Oscillator:
       AO = SMA Fast - SMA Slow

    Interpretation:
    - Positive AO: Fast momentum is greater than slow momentum (Bullish).
    - Negative AO: Fast momentum is less than slow momentum (Bearish).
    - Zero Line Crossover: Crossing above 0 is a buy signal, crossing below 0 is a sell signal.
    - Color coding (often used): Green if AO > Previous AO, Red if AO < Previous AO.

    Use Cases:
    - Momentum Confirmation: Used to confirm the strength of a trend.
    - Signal Generation: Zero line crossovers, "Saucer" signals (three histograms), and "Twin Peaks" (divergence).
    - Divergence: Divergence between price and AO can signal potential reversals.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    fast_window = int(parameters.get('fast_window', 5))
    slow_window = int(parameters.get('slow_window', 34))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')

    mid_price = (df[high_col] + df[low_col]) / 2

    sma_fast = mid_price.rolling(window=fast_window).mean()
    sma_slow = mid_price.rolling(window=slow_window).mean()

    ao_series = sma_fast - sma_slow
    ao_series.name = f'AO_{fast_window}_{slow_window}'

    return ao_series, [ao_series.name]
