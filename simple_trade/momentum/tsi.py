import pandas as pd


def tsi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the True Strength Index (TSI), a momentum oscillator developed by William Blau.
    It uses double smoothing of price changes to filter out noise and highlight trend strength.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - slow (int): Span for the first (slow) EMA smoothing of momentum. Default is 25.
            - fast (int): Span for the second (fast) EMA smoothing. Default is 13.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the TSI series and a list of column names.

    The True Strength Index is calculated as follows:

    1. Calculate Price Change (Momentum):
       Momentum = Close - Prev Close

    2. Calculate Double Smoothed Momentum (Numerator):
       First Smooth = EMA(Momentum, slow)
       Second Smooth = EMA(First Smooth, fast)

    3. Calculate Double Smoothed Absolute Momentum (Denominator):
       Abs Momentum = Abs(Momentum)
       First Smooth Abs = EMA(Abs Momentum, slow)
       Second Smooth Abs = EMA(First Smooth Abs, fast)

    4. Calculate TSI:
       TSI = 100 * (Double Smoothed Momentum / Double Smoothed Abs Momentum)

    Interpretation:
    - Range: -100 to +100.
    - Signal Line Crossovers: Can be used with a signal line (usually 7-12 period EMA of TSI) to generate buy/sell signals.
    - Centerline Crossovers: Crossing zero indicates a change in the overall trend direction.
    - Overbought/Oversold: Extremes can vary but generally > +25 or < -25 indicate strong trends.

    Use Cases:
    - Trend Direction: Positive TSI indicates uptrend, negative TSI indicates downtrend.
    - Divergence: Divergence between price and TSI can signal reversals.
    - Overbought/Oversold: Identifying potential exhaustion points.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    slow = int(parameters.get('slow', 25))
    fast = int(parameters.get('fast', 13))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]
    momentum = series.diff()

    def _double_ema(values: pd.Series) -> pd.Series:
        first = values.ewm(span=slow, adjust=False, min_periods=slow).mean()
        return first.ewm(span=fast, adjust=False, min_periods=slow + fast - 1).mean()

    smoothed_momentum = _double_ema(momentum)
    smoothed_abs_momentum = _double_ema(momentum.abs())

    tsi_series = 100 * smoothed_momentum / smoothed_abs_momentum.replace({0: pd.NA})
    tsi_series.name = f'TSI_{slow}_{fast}'

    columns_list = [tsi_series.name]
    return tsi_series, columns_list
