import numpy as np
import pandas as pd


def sri(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Stochastic RSI (StochRSI), a technical indicator used to measure the level of RSI relative to its high-low range over a set time period.
    It applies the Stochastic Oscillator formula to RSI values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - rsi_window (int): The period for the RSI calculation. Default is 14.
            - stoch_window (int): The lookback period for Stochastic calculation. Default is 14.
            - k_window (int): The smoothing period for %K. Default is 3.
            - d_window (int): The smoothing period for %D. Default is 3.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the StochRSI DataFrame (%K and %D) and a list of column names.

    The Stochastic RSI is calculated as follows:

    1. Calculate RSI:
       RSI = RSI(Close, rsi_window)

    2. Calculate StochRSI (%K Raw):
       StochRSI = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)
       (Where Lowest/Highest RSI are over stoch_window)

    3. Calculate %K:
       %K = SMA(StochRSI, k_window) * 100

    4. Calculate %D:
       %D = SMA(%K, d_window)

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Values above 80 are considered overbought.
    - Oversold: Values below 20 are considered oversold.
    - Sensitivity: StochRSI is much more sensitive than RSI and reaches extremes more frequently.

    Use Cases:
    - Entry Signals: Crossing above 20 (Buy) or below 80 (Sell).
    - Crossovers: %K crossing above %D is bullish; below is bearish.
    - Trend Confirmation: Identifying short-term overbought/oversold conditions within a trend.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    rsi_window = int(parameters.get('rsi_window', 14))
    stoch_window = int(parameters.get('stoch_window', 14))
    k_window = int(parameters.get('k_window', 3))
    d_window = int(parameters.get('d_window', 3))
    close_col = columns.get('close_col', 'Close')

    close = df[close_col]

    # Calculate RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1 / rsi_window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))

    # Calculate StochRSI
    lowest_rsi = rsi_val.rolling(window=stoch_window).min()
    highest_rsi = rsi_val.rolling(window=stoch_window).max()

    stoch_rsi = (rsi_val - lowest_rsi) / (highest_rsi - lowest_rsi)

    # Smoothing (K and D)
    k_val = (stoch_rsi * 100).rolling(window=k_window).mean()  # Smooth K
    d_val = k_val.rolling(window=d_window).mean()  # Smooth D

    k_col = f'SRI_K_{rsi_window}_{stoch_window}'
    d_col = f'SRI_D_{d_window}'

    result = pd.DataFrame({
        k_col: k_val,
        d_col: d_val
    })

    return result, list(result.columns)
