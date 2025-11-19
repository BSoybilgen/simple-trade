import pandas as pd


def kst(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Know Sure Thing (KST) momentum indicator.

    Args:
        df (pd.DataFrame): Input price data containing close prices.
        parameters (dict, optional): Calculation parameters.
            - roc_periods (iterable): Lookback periods for the ROC calculations. Default is (10, 15, 20, 30).
            - ma_periods (iterable): Smoothing window for each ROC series. Default is (10, 10, 10, 15).
            - weights (iterable): Weights applied to each smoothed ROC. Default is (1, 2, 3, 4).
            - signal (int): Window length for the signal line. Default is 9.
        columns (dict, optional): Column overrides.
            - close_col (str): Column name for closing prices. Default is 'Close'.

    Returns:
        tuple: (DataFrame with KST and signal line, [column names])
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
