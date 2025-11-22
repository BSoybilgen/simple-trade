import pandas as pd


def kst(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Know Sure Thing (KST), a momentum oscillator developed by Martin Pring 
    that combines multiple Rate of Change (ROC) timeframes into a single indicator.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - roc_periods (iterable): Lookback periods for the ROC calculations. Default is (10, 15, 20, 30).
            - ma_periods (iterable): Smoothing window for each ROC series. Default is (10, 10, 10, 15).
            - weights (iterable): Weights applied to each smoothed ROC. Default is (1, 2, 3, 4).
            - signal (int): Window length for the signal line. Default is 9.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing a DataFrame (with KST and Signal columns) and a list of column names.

    The KST is calculated in the following steps:

    1. Calculate Rate of Change (ROC) for four different periods:
       ROC1, ROC2, ROC3, ROC4

    2. Smooth each ROC with a Simple Moving Average (SMA):
       RCMA1 = SMA(ROC1, ma_period1)
       RCMA2 = SMA(ROC2, ma_period2)
       RCMA3 = SMA(ROC3, ma_period3)
       RCMA4 = SMA(ROC4, ma_period4)

    3. Calculate KST Line (Weighted Sum):
       KST = (RCMA1 * W1) + (RCMA2 * W2) + (RCMA3 * W3) + (RCMA4 * W4)

    4. Calculate Signal Line:
       Signal = SMA(KST, signal_period)

    Interpretation:
    - KST crossing above Signal Line: Bullish signal.
    - KST crossing below Signal Line: Bearish signal.
    - Zero Line Crossover: KST crossing zero confirms trend direction (Positive = Uptrend, Negative = Downtrend).

    Use Cases:
    - Trend Confirmation: Validating the strength and direction of a trend using multiple timeframes.
    - Signal Generation: Crossovers of KST and its signal line.
    - Divergence: Divergence between price and KST can signal reversals.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    roc_periods = parameters.get('roc_periods', (10, 15, 20, 30))
    ma_periods = parameters.get('ma_periods', (10, 10, 10, 15))
    weights = parameters.get('weights', (1, 2, 3, 4))
    signal_period = int(parameters.get('signal', 9))
    close_col = columns.get('close_col', 'Close')

    roc_periods = [int(p) for p in roc_periods]
    ma_periods = [int(p) for p in ma_periods]
    weights = [float(w) for w in weights]

    if not (len(roc_periods) == len(ma_periods) == len(weights)):
        raise ValueError('roc_periods, ma_periods, and weights must have the same length')

    close = df[close_col]
    smoothed_rocs = []

    for roc_period, ma_period in zip(roc_periods, ma_periods):
        roc_series = close.pct_change(periods=roc_period) * 100
        smoothed = roc_series.rolling(window=ma_period).mean()
        smoothed_rocs.append(smoothed)

    kst_series = sum(weight * series for weight, series in zip(weights, smoothed_rocs))
    kst_series.name = 'KST'

    signal_series = kst_series.rolling(window=signal_period).mean()
    signal_series.name = f'KST_Signal_{signal_period}'

    result = pd.DataFrame({kst_series.name: kst_series, signal_series.name: signal_series})
    result.index = df.index

    columns_list = list(result.columns)
    return result, columns_list
