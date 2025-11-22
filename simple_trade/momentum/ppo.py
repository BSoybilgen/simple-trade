import pandas as pd


def ppo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Percentage Price Oscillator (PPO), a momentum indicator similar to MACD.
    It measures the difference between two moving averages as a percentage of the larger moving average.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - fast_window (int): The window size for the fast EMA. Default is 12.
            - slow_window (int): The window size for the slow EMA. Default is 26.
            - signal_window (int): The window size for the signal line EMA. Default is 9.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the PPO DataFrame (PPO, Signal, Hist) and a list of column names.

    The Percentage Price Oscillator is calculated as follows:

    1. Calculate Fast and Slow EMAs:
       Fast EMA = EMA(Close, fast_window)
       Slow EMA = EMA(Close, slow_window)

    2. Calculate PPO Line:
       PPO = ((Fast EMA - Slow EMA) / Slow EMA) * 100

    3. Calculate Signal Line:
       Signal = EMA(PPO, signal_window)

    4. Calculate Histogram:
       Histogram = PPO - Signal

    Interpretation:
    - Same as MACD but in percentage terms, allowing comparison across securities with different prices.
    - Crossovers: PPO crossing above Signal is bullish; below is bearish.
    - Zero Line: Crossing zero signals a change in trend direction (Short-term average crossing Long-term average).

    Use Cases:
    - Asset Comparison: comparing momentum between assets of different prices.
    - Trend Confirmation: Validating trend direction and strength.
    - Divergence: Identifying potential reversals.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    fast_window = int(parameters.get('fast_window', 12))
    slow_window = int(parameters.get('slow_window', 26))
    signal_window = int(parameters.get('signal_window', 9))
    close_col = columns.get('close_col', 'Close')

    close = df[close_col]

    ema_fast = close.ewm(span=fast_window, adjust=False).mean()
    ema_slow = close.ewm(span=slow_window, adjust=False).mean()

    ppo_line = ((ema_fast - ema_slow) / ema_slow) * 100
    signal_line = ppo_line.ewm(span=signal_window, adjust=False).mean()
    histogram = ppo_line - signal_line

    ppo_col = f'PPO_{fast_window}_{slow_window}'
    sig_col = f'PPO_SIG_{signal_window}'
    hist_col = 'PPO_HIST'

    result = pd.DataFrame({
        ppo_col: ppo_line,
        sig_col: signal_line,
        hist_col: histogram
    })

    return result, list(result.columns)
