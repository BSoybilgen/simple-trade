import pandas as pd


def dpo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Detrended Price Oscillator (DPO), an indicator designed to remove trend 
    from price and make it easier to identify cycles and overbought/oversold levels.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the SMA. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the DPO series and a list of column names.

    The Detrended Price Oscillator is calculated as follows:

    1. Calculate the Simple Moving Average (SMA):
       SMA = SMA(Close, window)

    2. Calculate the Displacement:
       Displacement = (window / 2) + 1

    3. Calculate DPO:
       DPO = Close - SMA(shifted by Displacement)
       (Note: The formula effectively compares the current price to a past SMA value).

    Interpretation:
    - Positive DPO: Price is above the displaced moving average (Bullish/Overbought).
    - Negative DPO: Price is below the displaced moving average (Bearish/Oversold).
    - Zero Line Crossings: Can signal a change in the short-term trend or cycle.

    Use Cases:
    - Cycle Identification: Isolate short-term cycles by removing long-term trends.
    - Divergence: Identify potential reversals when price and DPO diverge.
    - Overbought/Oversold: Identify extremes within the cycle.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 20))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]
    sma = series.rolling(window=window, min_periods=window).mean()

    displacement = (window // 2) + 1
    displaced_sma = sma.shift(displacement)

    dpo_values = series - displaced_sma
    dpo_values.name = f'DPO_{window}'

    columns_list = [dpo_values.name]
    return dpo_values, columns_list
