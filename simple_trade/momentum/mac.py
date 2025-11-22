import pandas as pd

def mac(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Moving Average Convergence Divergence (MACD), Signal Line, and Histogram.
    It is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window_slow (int): The window size for the slower EMA. Default is 26.
            - window_fast (int): The window size for the faster EMA. Default is 12.
            - window_signal (int): The window size for the signal line EMA. Default is 9.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the MACD DataFrame (MACD, Signal, Hist) and a list of column names.

    Calculation Steps:
    1. Calculate the Fast EMA:
       Fast EMA = EMA(Close, window_fast)

    2. Calculate the Slow EMA:
       Slow EMA = EMA(Close, window_slow)

    3. Calculate the MACD Line:
       MACD Line = Fast EMA - Slow EMA

    4. Calculate the Signal Line:
       Signal Line = EMA(MACD Line, window_signal)

    5. Calculate the MACD Histogram:
       Histogram = MACD Line - Signal Line

    Interpretation:
    - Crossovers: MACD crossing above Signal Line is bullish; crossing below is bearish.
    - Zero Line: MACD crossing above zero suggests uptrend (Fast EMA > Slow EMA); below zero suggests downtrend.
    - Divergence: Divergence between Price and MACD/Histogram suggests waning momentum and potential reversal.

    Use Cases:
    - Trend Identification: Confirming trend direction and strength.
    - Entry/Exit Signals: Signal line crossovers are common entry/exit points.
    - Momentum Measurement: The width of the histogram indicates the speed of price movement.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window_slow = int(parameters.get('window_slow', 26))
    window_fast = int(parameters.get('window_fast', 12))
    window_signal = int(parameters.get('window_signal', 9))
    close_col = columns.get('close_col', 'Close')
    
    series = df[close_col]
    ema_fast = series.ewm(span=window_fast, adjust=False).mean()
    ema_slow = series.ewm(span=window_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=window_signal, adjust=False).mean()
    histogram = macd_line - signal_line
    # Return DataFrame for multi-output indicators
    df_macd = pd.DataFrame({
        f'MACD_{window_fast}_{window_slow}': macd_line,
        f'Signal_{window_signal}': signal_line,
        f'Hist_{window_fast}_{window_slow}_{window_signal}': histogram
    })
    # Ensure index is passed explicitly, just in case
    df_macd.index = series.index
    columns_list = list(df_macd.columns)
    return df_macd, columns_list